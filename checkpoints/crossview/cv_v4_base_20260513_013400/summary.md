# Run Summary — cv_v4_base_20260513_013400

## Config
- **Backbone:** `convnext_base.fb_in22k`
- **Embed dim:** 0
- **Image size:** 224
- **Batch size:** 64
- **Epochs:** 36
- **LR:** 0.0001 (weight_decay=0.0001)
- **Temperature:** init=0.07, learnable=True
- **Label smoothing:** 0.1
- **Hard negatives:** enabled=True, gps_epochs=4, pool_size=32, dss_refresh_every=1
- **Eval directions:** ['g2s', 's2g']
- **Eval metrics:** recall_ks=[1, 5, 10], map_k=1000

## Runtime
- Total wall time: **384.3 min** (23059 s)

## Best Epoch
- Epoch: **36**  (selected on `g2s_recall@1`)
- Score: **0.0763**
- **G2S**: recall@1=0.0763, recall@5=0.1902, recall@10=0.2457, map@1000=0.1334
- **S2G**: recall@1=0.0400, recall@5=0.1130, recall@10=0.1680, map@1000=0.0212

## Final Epoch
- Epoch: **36**
- **G2S**: recall@1=0.0763, recall@5=0.1902, recall@10=0.2457, map@1000=0.1334
- **S2G**: recall@1=0.0400, recall@5=0.1130, recall@10=0.1680, map@1000=0.0212

## Files
- `config.yaml` — resolved config used for this run
- `train_curves.csv` — per-epoch training stats
- `eval_curves.csv` — per-epoch evaluation metrics
- `best.pt`, `last.pt` — checkpoints
- `best_metrics.json`, `final_metrics.json` — metrics snapshots
