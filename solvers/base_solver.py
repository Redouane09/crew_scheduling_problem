from abc import ABC, abstractmethod
import json
import time

class BaseSolver(ABC):
    """Abstract base class for all solvers in the crew scheduling benchmark"""
    
    def __init__(self, name):
        self.name = name
        self.solve_time = 0.0
        self.objective_value = None
    
    @abstractmethod
    def solve(self, instance_data, time_limit=300.0):
        """
        Solve the crew scheduling problem
        
        Args:
            instance_data: Dictionary containing problem instance
            time_limit: Maximum solving time in seconds
            
        Returns:
            solution: Dictionary with solution details
        """
        pass
    
    @abstractmethod
    def get_solver_stats(self):
        """Return solver statistics"""
        pass
    
    def load_instance(self, filename):
        """Load instance from JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)
    
    def save_solution(self, solution, filename):
        """Save solution to JSON file"""
        with open(filename, 'w') as f:
            json.dump(solution, f, indent=2)