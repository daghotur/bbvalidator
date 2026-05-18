import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-Head Attention Pooling
# ---------------------------------------------------------------------------
class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads

        self.score_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # FIX: было d_model → d_model
            nn.Tanh(),
            nn.Linear(d_model // 2, num_heads),
        )

        self.out_proj = nn.Linear(d_model * num_heads, d_model)

    def forward(
            self,
            x: torch.Tensor,  # [B, N, d_model]
            mask: torch.Tensor,  # [B, N] bool  —  True = valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score_net(x)  # [B, N, H]
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)  # [B, N, H]

        pooled = torch.einsum("bnh,bnd->bhd", weights, x)  # [B, H, d]
        pooled_flat = pooled.flatten(start_dim=1)  # [B, H*d]

        return self.out_proj(pooled_flat), weights


# ---------------------------------------------------------------------------
# Multi-Task Heads
# ---------------------------------------------------------------------------
class MLPHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProteinMultiTaskHeads(nn.Module):
    """
    Пять независимых голов для регуляризации энкодера:
      fold_logit   — бинарная: foldable vs decoy
      rmsd         — регрессия RMSD (в log1p-пространстве)
      steric       — регрессия стерических клэшей
      hbond        — регрессия числа водородных связей
      failure_mode — классификация: 0=Ok, 1=Clash, 2=Core, 3=Loop
    """

    def __init__(self, d_model: int, dropout: float = 0.15):
        super().__init__()
        self.fold_head = MLPHead(d_model, 1, dropout)
        self.rmsd_head = MLPHead(d_model, 1, dropout)
        self.steric_head = MLPHead(d_model, 1, dropout)
        self.hbond_head = MLPHead(d_model, 1, dropout)
        self.failure_mode_head = MLPHead(d_model, 6, dropout)

    def forward(self, protein_repr: torch.Tensor) -> dict:
        return {
            "fold_logit": self.fold_head(protein_repr).squeeze(-1),  # [B]
            "rmsd": self.rmsd_head(protein_repr).squeeze(-1),  # [B]
            "steric": self.steric_head(protein_repr).squeeze(-1),  # [B]
            "hbond": self.hbond_head(protein_repr).squeeze(-1),  # [B]
            "failure_mode": self.failure_mode_head(protein_repr),  # [B, 4]
        }


# ---------------------------------------------------------------------------
# Динамический лосс (Homoscedastic Task Uncertainty)
# ---------------------------------------------------------------------------
class DynamicMultiTaskLoss(nn.Module):
    """
    Автоматическое взвешивание задач по Kendall & Gal (2018):

        L = Σ_i [ L_i / (2σ_i²) + log σ_i ]
          = Σ_i [ 0.5 * exp(-s_i) * L_i  +  0.5 * s_i ]

    где s_i = log(σ_i²) — обучаемые параметры.
    """

    def __init__(
            self,
            num_tasks: int = 5,
            failure_class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        # Обучаемые лог-дисперсии, по одной на задачу
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

        # FIX [4]: веса классов для компенсации дисбаланса failure_mode.
        # Пример вычисления перед обучением:
        #   counts = torch.tensor([n_ok, n_clash, n_core, n_loop], dtype=float)
        #   weights = (1.0 / counts); weights /= weights.sum()
        if failure_class_weights is not None:
            self.register_buffer("failure_class_weights", failure_class_weights)
        else:
            self.failure_class_weights = None

    def forward(self, preds: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        # squeeze(-1) и .float() делают код устойчивым к лейблам формы [B] и [B, 1]
        fold_label = batch["label"].float().view(-1)

        # 1. Бинарная классификация фолдинга (главная задача)
        loss_fold = F.binary_cross_entropy_with_logits(
            preds["fold_logit"], fold_label
        )

        # 2. RMSD — log1p стабилизирует хвост для разрушенных структур
        rmsd_target = torch.log1p(batch["rmsd_target"].float())
        loss_rmsd = F.mse_loss(preds["rmsd"], rmsd_target)

        # 3. Физические прокси
        loss_steric = F.mse_loss(preds["steric"], batch["steric_target"].float())
        loss_hbond = F.mse_loss(preds["hbond"], batch["hbond_target"].float())

        # 4. Тип ошибки структуры (с поддержкой class weights)
        loss_fail = F.cross_entropy(
            preds["failure_mode"],
            batch["failure_mode_label"].long(),
            weight=self.failure_class_weights,
        )

        losses = torch.stack([loss_fold, loss_rmsd, loss_steric, loss_hbond, loss_fail])

        # FIX [1]: правильная формула Kendall — множитель 0.5
        precision = torch.exp(-self.log_vars)
        total_loss = torch.sum(0.5 * precision * losses + 0.5 * self.log_vars)

        return total_loss, {
            "loss_fold": loss_fold.item(),
            "loss_rmsd": loss_rmsd.item(),
            "loss_steric": loss_steric.item(),
            "loss_hbond": loss_hbond.item(),
            "loss_fail": loss_fail.item(),
            "weights": precision.detach().cpu().numpy(),
        }


# ---------------------------------------------------------------------------
# Сборка модели
# ---------------------------------------------------------------------------
class ProteinScoreModel(nn.Module):
    def __init__(self, frontend, encoder, pooler, heads):
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.pooler = pooler
        self.heads = heads

    def compute_features(
            self, coords: torch.Tensor, mask: torch.Tensor
    ) -> dict:
        return self.frontend(coords, mask)

    def encode_features(
            self, features: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Граф + трансформер + пулинг. Повторяется mc_runs раз при MC-Dropout."""
        node_encoded = self.encoder(features)
        protein_repr, attn_weights = self.pooler(node_encoded, features["mask"])
        return protein_repr, attn_weights

    def predict_from_repr(self, protein_repr: torch.Tensor) -> dict:
        return self.heads(protein_repr)

    def forward(self, coords: torch.Tensor, mask: torch.Tensor) -> dict:
        features = self.compute_features(coords, mask)
        protein_repr, attn_weights = self.encode_features(features)
        preds = self.predict_from_repr(protein_repr)
        preds["attn_weights"] = attn_weights
        return preds


# ---------------------------------------------------------------------------
# Стадия 5: MC-Dropout инференс
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_with_uncertainty(
        model: ProteinScoreModel,
        coords: torch.Tensor,
        mask: torch.Tensor,
        mc_runs: int = 16,
) -> dict:
    """
    Оценка неопределённости через MC-Dropout.

    Ключевое разделение:
      • compute_features — физика, 1 раз (без дропаута)
      • encode_features  — граф + трансформер, mc_runs раз (с дропаутом)

    Параметры
    ----------
    model   : обученная ProteinScoreModel
    coords  : [B, N, 3]  — координаты Cα
    mask    : [B, N]  bool — True = валидный остаток
    mc_runs : число форвард-пассов для оценки дисперсии (8–32 обычно достаточно)

    Возвращает dict:
      p_foldable  : [B]  — усреднённая вероятность успешного фолдинга
      uncertainty : [B]  — дисперсия предсказаний (мера неопределённости)
    """
    model.eval()

    # 1. Тяжёлые физические признаки — только один раз
    features = model.compute_features(coords, mask)

    # 2. Сохраняем состояние обучения ВСЕХ модулей
    saved_states: dict[nn.Module, bool] = {
        m: m.training for m in model.modules()
    }

    # 3. Включаем dropout только там, где нужна неопределённость.
    def _enable_dropout(m: nn.Module) -> None:
        if isinstance(m, nn.Dropout):
            m.train()

    model.encoder.apply(_enable_dropout)
    model.heads.apply(_enable_dropout)

    probs: list[torch.Tensor] = []

    try:
        for _ in range(mc_runs):
            protein_repr, _ = model.encode_features(features)
            preds = model.predict_from_repr(protein_repr)
            probs.append(torch.sigmoid(preds["fold_logit"]))  # [B]

    finally:
        for m, state in saved_states.items():
            m.training = state

    stacked = torch.stack(probs, dim=0)  # [mc_runs, B]
    mean_prob = stacked.mean(dim=0)  # [B]
    var_prob = stacked.var(dim=0)  # [B]

    return {
        "p_foldable": mean_prob,
        "uncertainty": var_prob,
    }
