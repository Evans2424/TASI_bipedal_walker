# Cleanup Plan - Final Bridge Walker Training

## Files to KEEP (Working Solution)
1. **bridge_shaped_wrapper.py** - The working solution with LIDAR detection
2. **configs/sac_bridge_shaped_gpu.yaml** - Configuration for bridge-shaped training
3. **train_custom_walker.py** - Main training script (will simplify)
4. **custom_walker.py** - Custom environment with bridges
5. **visualize_custom_walker.py** - Visualization script

## Files to REMOVE (Failed Experiments)
1. elite_hardcore_bridge_wrapper.py - V1 (failed - exploited)
2. elite_hardcore_bridge_wrapper_v2.py - V2 (failed - still exploited)
3. bridge_optimized_wrapper.py - Soft penalties (failed - stuck at 90 steps)
4. configs/sac_elite_hardcore_gpu.yaml - Not used for bridges
5. configs/sac_bridge_optimized_gpu.yaml - Failed approach

## Simplifications Needed

### train_custom_walker.py
- Remove support for use_elite_hardcore (not for bridges)
- Remove support for use_bridge_optimized (failed)
- Keep ONLY use_bridge_shaped
- Simplify make_env() function
- Cleaner configuration parsing

### Final Clean Structure
```
bipedal_walker/
├── custom_walker.py                    # Custom environment with bridges
├── bridge_shaped_wrapper.py            # WORKING wrapper (LIDAR-based)
├── train_custom_walker.py              # Simplified training script
├── visualize_custom_walker.py          # Visualization
├── configs/
│   └── sac_bridge_shaped_gpu.yaml     # Final working config
└── experiments/
    ├── checkpoints/
    └── logs/
```

## Action Items
1. Remove failed wrapper files
2. Simplify train_custom_walker.py (remove failed options)
3. Update visualize script to use bridge_shaped wrapper
4. Create final README with working approach only
5. Archive/backup failed experiments for reference
