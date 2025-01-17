from concurrent.futures import ProcessPoolExecutor
from rich.live import Live
from rich.table import Table
from rich import box
import time


class OptimizationProgress:
    """Generic progress tracker for optimization tasks."""
    def __init__(self, title, param_names, budget, extra_params=None):
        """
        Args:
            title (str): Title for the progress table
            param_names (list): List of parameter names to display
            budget (int): Total optimization budget
            extra_params (dict, optional): Additional parameters to display in title
        """
        self.title = title
        self.param_names = param_names
        self.extra_params = extra_params or {}
        self.best_value = float('inf')
        self.best_params = None
        self.current_budget = budget
        self.running_jobs = 0
        self.current_evaluations = []
        self.start_time = time.time()
        self.total_evaluations = 0
    
    def create_table(self):
        # Create title with extra parameters if provided
        title = self.title
        if self.extra_params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.extra_params.items())
            title = f"{title} ({params_str})"
        
        table = Table(title=title, box=box.ROUNDED)
        
        # Status columns
        table.add_column("Time", justify="right", style="cyan", width=8)
        table.add_column("Budget", justify="right", style="cyan", width=8)
        table.add_column("Jobs", justify="right", style="cyan", width=6)
        table.add_column("s/iter", justify="right", style="cyan", width=8)
        table.add_column("Best", justify="right", style="green", width=10)
        
        # Parameter columns for best result
        for param in self.param_names:
            table.add_column(f"Best {param}", justify="right", style="yellow", width=10)
        
        # Current evaluation columns
        for param in self.param_names:
            table.add_column(param, justify="right", style="magenta", width=10)
        table.add_column("Cost", justify="right", style="red", width=10)

        # Add main status row
        elapsed = time.time() - self.start_time
        s_per_iter = "-" if self.total_evaluations == 0 else f"{elapsed/self.total_evaluations:.2f}"
        
        best_row = [
            f"{elapsed:.2f}s",
            str(self.current_budget),
            str(self.running_jobs),
            s_per_iter,
            f"{self.best_value:.6f}",
        ]
        
        # Add best parameters
        if self.best_params:
            best_row.extend([f"{self.best_params[param]:.6f}" for param in self.param_names])
        else:
            best_row.extend(["-"] * len(self.param_names))
        
        # Add empty cells for current evaluation columns
        best_row.extend(["-"] * (len(self.param_names) + 1))
        table.add_row(*best_row)
        
        # Add current evaluations
        if self.current_evaluations:
            sorted_evals = sorted(self.current_evaluations, key=lambda x: x[1])
            for params, value in sorted_evals:
                row = ["-"] * (5 + len(self.param_names))  # Status and best params columns
                row.extend([f"{params[param]:.4f}" for param in self.param_names])
                row.append(f"{value:.6f}")
                table.add_row(*row)
        
        return table
    
    def update(self, value, params=None, budget=None, running_jobs=None):
        """Update the progress tracker with new information."""
        if value is not None and value < self.best_value:
            self.best_value = value
            self.best_params = params
        if budget is not None:
            self.current_budget = budget
        if running_jobs is not None:
            self.running_jobs = running_jobs
        if params is not None and value is not None:
            self.current_evaluations.append((params, value))
            self.total_evaluations += 1
            if len(self.current_evaluations) > self.running_jobs:
                self.current_evaluations = self.current_evaluations[-self.running_jobs:]

def evaluate_candidate(candidate, objective_fn, **kwargs):
    """
    Generic function to evaluate a candidate solution.
    
    Args:
        candidate: Nevergrad candidate
        objective_fn: Function to optimize
        **kwargs: Additional arguments to pass to objective_fn
    """
    try:
        value = objective_fn(**kwargs, **candidate.kwargs)
        return candidate, value
    except Exception as e:
        print(f"Error evaluating candidate: {e}")
        return candidate, float('inf')

def run_optimization_with_progress(optimizer, 
                                   objective_fn, 
                                   param_names, 
                                   title, 
                                   budget, 
                                   num_workers, 
                                   show_live=True, 
                                   **kwargs):
    """
    Generic optimization runner with progress tracking.
    
    Args:
        optimizer: Nevergrad optimizer instance
        objective_fn: Function to optimize
        param_names: List of parameter names
        title: Title for the progress display
        budget: Total optimization budget
        num_workers: Number of parallel workers
        show_live (bool): Whether to show the live progress table
        **kwargs: Additional arguments to pass to objective_fn
    
    Returns:
        tuple: (recommendation, progress) containing the optimizer's final recommendation 
        and the progress tracker
    """
    progress = OptimizationProgress(title, param_names, budget, kwargs)
    
    def run_optimization():
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            remaining_budget = budget
            
            while remaining_budget > 0:
                candidates = []
                for _ in range(min(num_workers, remaining_budget)):
                    try:
                        candidates.append(optimizer.ask())
                    except Exception as e:
                        print(f"Error asking for candidate: {e}")
                        continue
                
                if not candidates:
                    break
                
                progress.update(None, running_jobs=len(candidates), budget=remaining_budget)
                
                futures = [
                    executor.submit(evaluate_candidate, c, objective_fn, **kwargs)
                    for c in candidates
                ]
                
                for future in futures:
                    try:
                        candidate, value = future.result()
                        optimizer.tell(candidate, value)
                        progress.update(value, candidate.kwargs)
                        remaining_budget -= 1
                    except Exception as e:
                        print(f"Error processing result: {e}")
                        continue
                
                if show_live:
                    time.sleep(0.1)  # Small delay to prevent excessive updates
    
    if show_live:
        with Live(progress.create_table(), refresh_per_second=2) as live:
            run_optimization()
            live.update(progress.create_table())
    else:
        run_optimization()
    
    return optimizer.recommend(), progress
