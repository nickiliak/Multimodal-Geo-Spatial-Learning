# Run Summary — cv_v2_20260419_220021

## Config
- **Backbone:** `convnext_tiny.fb_in22k`
- **Embed dim:** 0
- **Image size:** 224
- **Batch size:** 64
- **Epochs:** 20
- **LR:** 0.0001 (weight_decay=0.0001)
- **Temperature:** init=0.07, learnable=True
- **Label smoothing:** 0.0
- **Hard negatives:** enabled=True, gps_epochs=3, pool_size=32, dss_refresh_every=1
- **Eval directions:** ['g2s', 's2g']
- **Eval metrics:** recall_ks=[1, 5, 10], map_k=1000

## Runtime
- Total wall time: **689.6 min** (41378 s)

## Best Epoch
- Epoch: **20**  (selected on `g2s_recall@1`)
- Score: **0.0433**
- **G2S**: recall@1=0.0433, recall@5=0.1261, recall@10=0.1779, map@1000=0.0888
- **S2G**: recall@1=0.0290, recall@5=0.0770, recall@10=0.1170, map@1000=0.0147

## Final Epoch
- Epoch: **20**
- **G2S**: recall@1=0.0433, recall@5=0.1261, recall@10=0.1779, map@1000=0.0888
- **S2G**: recall@1=0.0290, recall@5=0.0770, recall@10=0.1170, map@1000=0.0147

## Files
- `config.yaml` — resolved config used for this run
- `train_curves.csv` — per-epoch training stats
- `eval_curves.csv` — per-epoch evaluation metrics
- `best.pt`, `last.pt` — checkpoints
- `best_metrics.json`, `final_metrics.json` — metrics snapshots
