# Cubic Graph Search Algorithm Analysis Report

## Experiment Overview
- **Cube Size**: 5x5x5 (125 nodes)
- **Search Space**: 3D lattice with 6-connectivity
- **Start Position**: (0,0,0) - Yellow-Orange-Blue corner
- **Goal Position**: (4,4,4) - Green-White-Red corner

## Algorithm Performance Comparison

### Breadth-First Search
- **Status**: ✓ Success
- **Path Length**: 13 steps
- **Path Cost**: 12
- **Color Sequence**: Yellow-Orange-Blue → Orange-Blue → Blue → Blue → Gray → Gray → Gray → Gray → Gray → Gray...

### Depth-First Search
- **Status**: ✓ Success
- **Path Length**: 31 steps
- **Path Cost**: 30
- **Color Sequence**: Yellow-Orange-Blue → Yellow-Blue → Yellow → Gray → Gray → Orange → Yellow-Orange → Yellow-Orange → Yellow → Yellow...

### Uniform Cost Search
- **Status**: ✓ Success
- **Path Length**: 13 steps
- **Path Cost**: 12
- **Color Sequence**: Yellow-Orange-Blue → Yellow-Orange → Yellow-Orange → Yellow-Orange → Yellow-Orange-Red → Yellow-Red → Yellow-Red → Yellow-Red → Yellow-White-Red → White-Red...

### A* Search
- **Status**: ✓ Success
- **Path Length**: 13 steps
- **Path Cost**: 12
- **Color Sequence**: Yellow-Orange-Blue → Yellow-Orange → Yellow-Orange → Yellow-Orange → Yellow-Orange-Red → Yellow-Red → Yellow-Red → Yellow-Red → Yellow-White-Red → White-Red...

## Key Findings

1. **Most Efficient Algorithm**: Breadth-First Search
2. **Optimal Path Length**: 13 steps
3. **Success Rate**: 4/4 algorithms found solutions

---
Generated: Stanford AI Class Preparation Project
