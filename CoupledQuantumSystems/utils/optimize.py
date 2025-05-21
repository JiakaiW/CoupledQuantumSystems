"""Optimization utilities for quantum systems.

This module provides tools for optimizing parameters of quantum systems using parallel processing
and progress tracking. It includes a progress visualization system and optimization runners
that work with Nevergrad optimizers.
"""

from concurrent.futures import ProcessPoolExecutor
import time
from rich.live import Live
from rich.table import Table
from rich import box

class OptimizationProgress:
    """Tracks and displays the progress of optimization tasks.

    This class provides a rich text-based progress display for optimization tasks,
    showing current parameters, best results, and timing information.

    Attributes:
        title (str): Title for the progress table.
        param_names (list): List of parameter names to display.
        extra_params (dict): Additional parameters to display in title.
        best_value (float): Current best optimization value.
        best_params (dict): Parameters corresponding to the best value.
        current_budget (int): Remaining optimization budget.
        running_jobs (int): Number of currently running optimization jobs.
        current_evaluations (list): List of current evaluation results.
        start_time (float): Time when optimization started.
        total_evaluations (int): Total number of evaluations performed.
    """

    def __init__(self, title, param_names, budget, extra_params=None):
        """Initialize the progress tracker.

        Args:
            title (str): Title for the progress table.
            param_names (list): List of parameter names to display.
            budget (int): Total optimization budget.
            extra_params (dict, optional): Additional parameters to display in title.
                Defaults to None.

        Example:
            >>> progress = OptimizationProgress(
            ...     title="Quantum Gate Optimization",
            ...     param_names=["frequency", "amplitude"],
            ...     budget=1000,
            ...     extra_params={"system": "qubit"}
            ... )
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
        """Create a rich table displaying the current optimization progress.

        Returns:
            rich.table.Table: A formatted table showing optimization progress.
        """
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
        """Update the progress tracker with new information.

        Args:
            value (float, optional): New optimization value. If better than current best,
                updates the best value and parameters.
            params (dict, optional): Parameters corresponding to the new value.
            budget (int, optional): Updated remaining budget.
            running_jobs (int, optional): Updated number of running jobs.

        Example:
            >>> progress.update(
            ...     value=0.123,
            ...     params={"frequency": 1.0, "amplitude": 0.5},
            ...     budget=950,
            ...     running_jobs=4
            ... )
        """
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
    """Evaluate a candidate solution using the objective function.

    Args:
        candidate (nevergrad.parametrization.Parameter): Candidate solution to evaluate.
        objective_fn (callable): Function to optimize. Should accept the candidate's
            parameters as keyword arguments.
        **kwargs: Additional arguments to pass to objective_fn.

    Returns:
        tuple: (candidate, value) containing the evaluated candidate and its objective value.
            If evaluation fails, returns (candidate, float('inf')).

    Example:
        >>> def objective(frequency, amplitude):
        ...     return (frequency - 1.0)**2 + (amplitude - 0.5)**2
        >>> candidate = nevergrad.p.Instrumentation(frequency=1.1, amplitude=0.6)
        >>> result = evaluate_candidate(candidate, objective)
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
    """Run an optimization process with progress tracking and parallel evaluation.

    This function manages the optimization process, including parallel evaluation
    of candidates and progress visualization.

    Args:
        optimizer (nevergrad.optimization.Optimizer): Nevergrad optimizer instance.
        objective_fn (callable): Function to optimize. Should accept parameters
            as keyword arguments.
        param_names (list): List of parameter names to display in progress.
        title (str): Title for the progress display.
        budget (int): Total optimization budget (number of evaluations).
        num_workers (int): Number of parallel workers for evaluation.
        show_live (bool, optional): Whether to show the live progress table.
            Defaults to True.
        **kwargs: Additional arguments to pass to objective_fn.

    Returns:
        tuple: (recommendation, progress) containing:
            - recommendation: The optimizer's final recommendation
            - progress: The OptimizationProgress instance tracking the optimization

    Example:
        >>> optimizer = nevergrad.optimizers.OnePlusOne(
        ...     parametrization=nevergrad.p.Instrumentation(
        ...         frequency=nevergrad.p.Scalar(),
        ...         amplitude=nevergrad.p.Scalar()
        ...     )
        ... )
        >>> result, progress = run_optimization_with_progress(
        ...     optimizer=optimizer,
        ...     objective_fn=objective,
        ...     param_names=["frequency", "amplitude"],
        ...     title="Gate Optimization",
        ...     budget=1000,
        ...     num_workers=4
        ... )
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
