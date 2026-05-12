# Run Summary — cv_v2_base_20260422_230539

## Config
- **Backbone:** `convnext_base.fb_in22k`
- **Embed dim:** 0
- **Image size:** 224
- **Batch size:** 64
- **Epochs:** 35
- **LR:** 0.0001 (weight_decay=0.0001)
- **Temperature:** init=0.07, learnable=True
- **Label smoothing:** 0.0
- **Hard negatives:** enabled=True, gps_epochs=4, pool_size=32, dss_refresh_every=1
- **Eval directions:** ['g2s', 's2g']
- **Eval metrics:** recall_ks=[1, 5, 10], map_k=1000

## Runtime
- Total wall time: **513.4 min** (30805 s)

## Best Epoch
- Epoch: **30**  (selected on `g2s_recall@1`)
- Score: **0.1760**
- **G2S**: recall@1=0.1760, recall@5=0.3340, recall@10=0.4220, map@1000=0.2550
- **S2G**: recall@1=0.0520, recall@5=0.1440, recall@10=0.1870, map@1000=0.0266

## Final Epoch
- Epoch: **35**
- **G2S**: recall@1=0.1760, recall@5=0.3300, recall@10=0.4100, map@1000=0.2546
- **S2G**: recall@1=0.0530, recall@5=0.1520, recall@10=0.2000, map@1000=0.0267

## Files
- `config.yaml` — resolved config used for this run
- `train_curves.csv` — per-epoch training stats
- `eval_curves.csv` — per-epoch evaluation metrics
- `best.pt`, `last.pt` — checkpoints
- `best_metrics.json`, `final_metrics.json` — metrics snapshots
