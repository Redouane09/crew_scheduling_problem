# Integrated Crew Scheduling Solver Comparison

**Comprehensive implementation of integrated crew pairing and rostering with multiple solving approaches**

This repository provides a complete, faithful implementation of the integrated crew scheduling model from:

> Saemi, S., Komijan, A. R., Tavakkoli-Moghaddam, R., & Fallah, M. (2021). A new mathematical model to cover crew pairing and rostering problems simultaneously. *Journal of Engineering Research*, 9(2), 218-233.

## Key Features

✅ **Exact MIP Model**: All constraints (2)-(35) implemented precisely as described in Section 4 of the paper  
✅ **Faithful Metaheuristics**: GA and PSO implementations match Sections 5.1-5.2 exactly (parameters, operators, encoding)  
✅ **Advanced Heuristics**: Greedy and ALNS implementations for fast approximate solutions  
✅ **Hybrid ML-Optimization**: Novel combinations of GNN, L2O-pMINLP, and MOSEK  
✅ **Benchmark Instances**: 12 instances matching Table 1 of the paper (35-134 flights)  
✅ **Comprehensive Comparison**: Unified framework comparing 8 different solving approaches  

## Implemented Solvers

### 1. Exact Solver
- **MOSEK**: Exact MIP solver using Fusion API (constraints 2-35 from paper)

### 2. Metaheuristic Solvers
- **Genetic Algorithm (GA)**: Population-based evolutionary approach with custom operators
- **Particle Swarm Optimization (PSO)**: Swarm intelligence with velocity-position updates
- **Greedy Heuristic**: Fast constructive approach for baseline comparison
- **Adaptive Large Neighborhood Search (ALNS)**: Advanced neighborhood search with adaptive operator selection

### 3. Hybrid ML-Optimization Solvers
- **L2O-MOSEK**: Learning-to-Optimize with MLP branching predictor + MOSEK
- **GNN-MOSEK**: Graph Neural Network variable fixing + MOSEK
- **GNN-L2O-MOSEK**: Novel triple hybrid combining GNN + L2O-pMINLP + MOSEK

## Model Implementation Highlights

### Complete Constraint Set (Paper Equations 2-35)
- **Constraints (2)-(4)**: Flight coverage, one flight per round, min sit time (30 min)
- **Constraints (5)-(7)**: Min rest time (11 hrs), home base start/end requirements
- **Constraints (8)-(26)**: Duty continuity, flight integrity, working day relationships
- **Constraints (27)-(30)**: Uncovered flight detection and deadhead modeling
- **Constraints (31)-(35)**: FAA regulations (max 4 flights/duty, 20 flights/horizon, 8 hrs flying/duty, 40 hrs/horizon, 12 hrs elapsed/duty)

### Critical FAA Rules Implemented
- Maximum 4 flights per duty day (Constraint 33)
- Maximum 20 flights per planning horizon (Constraint 34)
- Minimum 30-minute sit time between flights (Constraint 4)
- Minimum 11-hour rest between duties (Constraint 5)

## Installation

```bash
# Clone repository
git clone https://github.com/Redouane09/gnn-l2o-csp.git
cd gnn-l2o-csp

# Install dependencies
pip install -r requirements.txt

# For MOSEK-based solvers, obtain free academic license
# Visit: https://www.mosek.com/products/academic-licenses/
# Place license in: ~/.mosek/mosek.lic (Linux/Mac) or C:\Users\<Username>\mosek\mosek.lic (Windows)
```

## Quick Start

### Generate Benchmark Instances
```bash
python main.py generate
```

### Solve with Different Solvers

```bash
# Exact solver (MOSEK)
python main.py solve --solver mosek --input data/instances/instance_1.json --time-limit 300

# Genetic Algorithm
python main.py solve --solver ga --input data/instances/instance_1.json --time-limit 180

# Particle Swarm Optimization
python main.py solve --solver pso --input data/instances/instance_1.json --time-limit 180

# Greedy Heuristic
python main.py solve --solver greedy --input data/instances/instance_1.json

# Adaptive Large Neighborhood Search
python main.py solve --solver alns --input data/instances/instance_1.json --time-limit 180

# L2O-MOSEK Hybrid
python main.py solve --solver l2o-mosek --input data/instances/instance_1.json --time-limit 300

# GNN-MOSEK Hybrid
python main.py solve --solver gnn-mosek --input data/instances/instance_1.json --time-limit 300

# GNN-L2O-MOSEK Triple Hybrid (Novel)
python main.py solve --solver gnn-l2o-mosek --input data/instances/instance_1.json --time-limit 300
```

### Comprehensive Solver Comparison

```bash
# Compare all solvers on a single instance
python comprehensive_comparison.py --input data/instances/instance_1.json --time-limit 180

# Compare on multiple instances
python comprehensive_comparison.py --input "data/instances/instance_*.json" --time-limit 120

# View results
# - CSV files in comparison_results/
# - PNG visualizations in comparison_results/
```

### Run Full Benchmark

```bash
# Run all solvers on all 12 benchmark instances
python benchmark/runner.py

# Run only metaheuristic solvers
python -c "from benchmark.runner import BenchmarkRunner; \
           runner = BenchmarkRunner(); \
           runner.run_all_solvers(['data/instances/instance_1.json'], \
                                 time_limit=180, solvers_to_run='heuristic')"

# Run only hybrid solvers
python -c "from benchmark.runner import BenchmarkRunner; \
           runner = BenchmarkRunner(); \
           runner.run_all_solvers(['data/instances/instance_1.json'], \
                                 time_limit=180, solvers_to_run='hybrid')"
```

## Solver Details

### MOSEK (Exact Solver)
- **Type**: Exact MIP solver
- **Strengths**: Guarantees optimality (within gap), handles all constraints precisely
- **Weaknesses**: May time out on large instances
- **Recommended for**: Small-medium instances (< 100 flights)

### Genetic Algorithm (GA)
- **Type**: Evolutionary metaheuristic
- **Parameters**: Population=200, Iterations=120, Crossover=0.6, Mutation=0.3
- **Strengths**: Good exploration, proven effectiveness on CSP
- **Weaknesses**: Slower convergence than PSO
- **Recommended for**: Medium instances

### Particle Swarm Optimization (PSO)
- **Type**: Swarm intelligence metaheuristic  
- **Parameters**: Swarm=180, Iterations=100, w=0.7, c1=2.1, c2=1.9
- **Strengths**: Fast convergence, good exploitation
- **Weaknesses**: May get stuck in local optima
- **Recommended for**: Quick solutions on all instance sizes

### Greedy Heuristic
- **Type**: Constructive heuristic
- **Strengths**: Very fast, simple to understand
- **Weaknesses**: Low solution quality
- **Recommended for**: Baseline comparison, initial solutions

### ALNS (Adaptive Large Neighborhood Search)
- **Type**: Destroy-repair metaheuristic
- **Operators**: 4 destroy (random, worst, route, time-window), 3 repair (greedy, regret, random)
- **Strengths**: Adaptive operator selection, effective for routing problems
- **Weaknesses**: Tuning required for best performance
- **Recommended for**: Large instances, diversified search

### L2O-MOSEK Hybrid
- **Type**: Machine Learning + Exact solver
- **Components**: MLP branching predictor + MOSEK
- **Strengths**: Learns branching heuristics, can improve MOSEK performance
- **Weaknesses**: Requires training data for full effectiveness
- **Recommended for**: When training data available

### GNN-MOSEK Hybrid
- **Type**: Graph Neural Network + Exact solver
- **Components**: GNN variable fixing + MOSEK
- **Strengths**: Exploits graph structure, reduces problem size
- **Weaknesses**: Requires training data, PyTorch dependency
- **Recommended for**: Problems with clear graph structure

### GNN-L2O-MOSEK Triple Hybrid (Novel)
- **Type**: Advanced hybrid combining three methodologies
- **Components**: GNN variable fixing + MLP branching + MOSEK
- **Strengths**: Combines advantages of all approaches, novel contribution
- **Weaknesses**: Most complex implementation, requires training
- **Recommended for**: Research, when maximum performance needed

## Performance Comparison

Expected performance characteristics (varies by instance):

| Solver | Solution Quality | Speed | Scalability | ML Required |
|--------|-----------------|-------|-------------|-------------|
| MOSEK | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ | No |
| GA | ★★★★☆ | ★★★☆☆ | ★★★★☆ | No |
| PSO | ★★★★☆ | ★★★★☆ | ★★★★☆ | No |
| Greedy | ★★☆☆☆ | ★★★★★ | ★★★★★ | No |
| ALNS | ★★★★☆ | ★★★☆☆ | ★★★★★ | No |
| L2O-MOSEK | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | Optional |
| GNN-MOSEK | ★★★★★ | ★★★☆☆ | ★★★☆☆ | Optional |
| GNN-L2O-MOSEK | ★★★★★ | ★★★☆☆ | ★★★☆☆ | Optional |

## Directory Structure

```
gnn-l2o-csp/
├── data/
│   ├── instance_generator.py          # Instance generation (Saemi et al. Table 1)
│   └── instances/                      # Pre-generated benchmark instances (12 instances)
│       ├── instance_1.json             # 35 flights, 2 days, 15 crew
│       ├── instance_2.json             # 38 flights, 2 days, 15 crew
│       └── ...                         # Up to instance_12.json (134 flights, 6 days, 50 crew)
├── model/
│   └── integrated_csp_model.py         # EXACT MIP model (constraints 2-35, no deviations)
├── solvers/
│   ├── base_solver.py                  # Unified solver interface
│   ├── mosek_solver.py                 # MOSEK exact MIP solver
│   ├── ga_solver.py                    # Genetic Algorithm (Saemi et al. Section 5.1)
│   ├── pso_solver.py                   # PSO (Saemi et al. Section 5.2)
│   ├── greedy_solver.py                # Constructive greedy heuristic
│   ├── alns_solver.py                  # Adaptive Large Neighborhood Search
│   ├── l2o_mosek_solver.py             # L2O-pMINLP + MOSEK hybrid
│   ├── gnn_mosek_solver.py             # GNN variable fixing + MOSEK
│   └── gnn_l2o_mosek_solver.py         # Novel GNN + L2O + MOSEK triple hybrid
├── ml_models/
│   ├── mlp_branching.py                # MLP for L2O branching decisions
│   └── gnn_fixing.py                   # GNN for variable fixing predictions
├── benchmark/
│   ├── runner.py                       # Unified benchmarking framework
│   ├── metrics.py                      # Standardized evaluation metrics
│   └── report_generator.py             # HTML report generation
├── main.py                             # CLI interface for solver comparison
├── comprehensive_comparison.py         # Comprehensive solver comparison script
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Output Files

### Solution Files
Each solver produces JSON solution files with:
- Objective value
- Crew assignments (flight-by-flight schedules)
- Unused crew (reserve crew)
- Uncovered flights
- Deadhead flights
- Solver statistics (time, gap, iterations, etc.)

### Comparison Results
The `comprehensive_comparison.py` script generates:
- **CSV files**: Detailed comparison metrics
- **PNG visualizations**: 
  - Objective value comparison
  - Computation time comparison
  - Flight coverage comparison
  - Optimality gap analysis

### Benchmark Reports
The benchmark runner generates:
- **HTML report**: Interactive comparison across all instances and solvers
- **Individual results**: JSON files for each solver-instance pair

## Citation

If you use this code in your research, please cite:

```bibtex
@article{saemi2021new,
  title={A new mathematical model to cover crew pairing and rostering problems simultaneously},
  author={Saemi, S. and Komijan, A. R. and Tavakkoli-Moghaddam, R. and Fallah, M.},
  journal={Journal of Engineering Research},
  volume={9},
  number={2},
  pages={218--233},
  year={2021}
}
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional metaheuristics (Tabu Search, Simulated Annealing)
- Enhanced ML model training with historical data
- Parallel solver execution
- Real-world instance integration
- Visualization enhancements

## License

This project is provided for academic and research purposes. Please respect the original paper's contributions and cite appropriately.

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.

## Acknowledgments

- Original MIP model: Saemi et al. (2021)
- MOSEK optimization software: MOSEK ApS
- Machine learning frameworks: scikit-learn, PyTorch, PyTorch Geometric
