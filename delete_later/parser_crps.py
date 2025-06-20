import argparse
import os

def parse_arguments():
    """
    Parse command line arguments and configuration parameters for CRPS MME and MSE MME.
    
    RUN SCRIPT WITH COOLTURTLES DIRECTORY AS YOUR CWD
    """
    parser = argparse.ArgumentParser(
        description='CRPS MME Hyperparameter Tuning Configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Path configurations
    parser.add_argument('--data-path', type=str, 
                        default=r"data\June_May_Dataset",
                        help='Path to the dataset directory')
    
    parser.add_argument('--tuner-path', type=str,
                        default=r"results\tuner\CRPS_MME_TUNER_RESULTS",
                        help='Path to the tuner results directory')
    
    parser.add_argument('--tensorboard-log-dir', type=str,
                        default=r"results\CRPS_MME_TENSORBOARD",
                        help='Path to tensorboard log directory')
    
    # Tuning variables
    parser.add_argument('--tuner-iterations', nargs='+', type=int, default=[1, 2],
                        help='Number of runs/iterations')
    
    parser.add_argument('--max-trials', type=int, default=18,
                        help='Maximum number of trials (30 for 1 execution, 15 for 2)')
    
    parser.add_argument('--execution-per-trial', type=int, default=2,
                        help='Number of executions per trial')
    
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    
    # Model architecture variables and hyperparameters
    parser.add_argument('--cycles', nargs='+', type=int, default=[1, 3, 6, 7, 9],
                        help='Cycles with cold stunning events in validation set')
    
    parser.add_argument('--lead-times', nargs='+', type=int, default=[12, 48, 96],
                        help='Lead times for prediction (12, 24, 48, 72, 96, 108, 120)')
    
    parser.add_argument('--hours-back', type=int, default=24,
                        help='Hours to look back for input features')
    
    parser.add_argument('--temperatures', nargs='+', type=float, default=[0.0],
                        help='Temperature perturbations list')
    
    parser.add_argument('--output-units', type=int, default=100,
                        help='Number of ensemble predictions')
    
    parser.add_argument('--units', nargs='+', type=int, default=[16, 32, 64, 100, 128, 256],
                        help='Number of units/neurons per layer')
    
    parser.add_argument('--activations', nargs='+', type=str, default=['relu', 'selu', 'leaky_relu'],
                        choices=['relu', 'selu', 'leaky_relu', 'tanh', 'sigmoid'],
                        help='Activation functions to try')
    
    parser.add_argument('--output-activation', type=str, default='linear',
                        help='Output layer activation function')
    
    parser.add_argument('--learning-rates', nargs='+', type=float, default=[0.01],
                        help='Learning rates to try')
    
    parser.add_argument('--optimizers', nargs='+', type=str, default=['adam'],
                        choices=['adam', 'adadelta', 'SGD'],
                        help='Optimizers to try')
    
    parser.add_argument('--kernel-regularizer', type=str, default='l2',
                        help='Kernel regularizer type')
    
    parser.add_argument('--objective', type=str, default='val_crps',
                        help='Optimization objective')
    
    parser.add_argument('--callback-monitor', type=str, default='val_loss',
                        help='Metric to monitor for callbacks')
    
    return parser.parse_args()

def setup_global_configuration():
    """
    Set up global configuration variables for the CRPS MME and MSE MME.
    All variables will be available globally with their original names.
    """
    global path_to_data, path_to_tuner, tensorboard_log_dir
    global tuner_iterations, max_trials, execution_per_trial, epochs
    global cycle_list, lead_time_list, hours_back, temperature_list, output_units
    global unit_list, activation_list, output_activation, learning_rate_list
    global optimizer_list, kernel_regularizer, obj, call_back_monitor, compute_times
    global loss_function, metrics
    
    args = parse_arguments()
    
    # Assign to global variables with original names
    # Paths
    path_to_data        = args.data_path
    path_to_tuner       = args.tuner_path
    tensorboard_log_dir = args.tensorboard_log_dir
    
    # Tuning variables
    tuner_iterations    = args.tuner_iterations
    max_trials          = args.max_trials
    execution_per_trial = args.execution_per_trial
    epochs              = args.epochs
    
    # Model architecture variables and hyperparameters
    cycle_list          = args.cycles
    lead_time_list      = args.lead_times
    hours_back          = args.hours_back
    temperature_list    = args.temperatures
    output_units        = args.output_units
    unit_list           = args.units
    activation_list     = args.activations
    output_activation   = args.output_activation
    learning_rate_list  = args.learning_rates
    optimizer_list      = args.optimizers
    kernel_regularizer  = args.kernel_regularizer
    obj                 = args.objective
    call_back_monitor   = args.callback_monitor
    
    # Initialize compute times dictionary
    compute_times = {}

    # Set loss function and metrics - these need to be imported/defined in your main script
    # You have a few options:
    # Option 1: Import them here (if they're available)
    
    loss_function = None
    metrics = None
    
    
    # Validate paths exist
    for path_name, path_value in [('path_to_data', path_to_data), 
                                  ('path_to_tuner', path_to_tuner), 
                                  ('tensorboard_log_dir', tensorboard_log_dir)]:
        if not os.path.exists(path_value):
            print(f"Warning: {path_name} ({path_value}) does not exist. Creating directory...")
            os.makedirs(path_value, exist_ok=True)


def get_all_variables():
    """
    Alternative approach: Return all variables as a tuple for unpacking.
    Usage: path_to_data, path_to_tuner, ... = get_all_variables()
    """
    args = parse_arguments()
    
    # Validate paths exist
    for path_name, path_value in [('data_path', args.data_path), 
                                  ('tuner_path', args.tuner_path), 
                                  ('tensorboard_log_dir', args.tensorboard_log_dir)]:
        if not os.path.exists(path_value):
            print(f"Warning: {path_name} ({path_value}) does not exist. Creating directory...")
            os.makedirs(path_value, exist_ok=True)
    
    # Return all variables in the same order as your original code
    return (
        # Paths
        args.data_path,           # path_to_data
        args.tuner_path,          # path_to_tuner
        args.tensorboard_log_dir, # tensorboard_log_dir
        
        # Tuning variables
        args.tuner_iterations,    # tuner_iterations
        args.max_trials,          # max_trials
        args.execution_per_trial, # execution_per_trial
        args.epochs,              # epochs
        
        # Model architecture variables and hyperparameters
        args.cycles,              # cycle_list
        args.lead_times,          # lead_time_list
        args.hours_back,          # hours_back
        args.temperatures,        # temperature_list
        args.output_units,        # output_units
        args.units,               # unit_list
        args.activations,         # activation_list
        args.output_activation,   # output_activation
        args.learning_rates,      # learning_rate_list
        args.optimizers,          # optimizer_list
        args.kernel_regularizer,  # kernel_regularizer
        args.objective,           # obj
        args.callback_monitor,    # call_back_monitor
        {}                        # compute_times (empty dict)
    )


def print_configuration():
    """Print the current global configuration for verification."""
    print("=" * 60)
    print("CRPS MME HYPERPARAMETER TUNING CONFIGURATION")
    print("=" * 60)
    
    print("\nPATHS:")
    print(f"  Data Path: {path_to_data}")
    print(f"  Tuner Path: {path_to_tuner}")
    print(f"  Tensorboard Log Dir: {tensorboard_log_dir}")
    
    print("\nTUNING PARAMETERS:")
    print(f"  Tuner Iterations: {tuner_iterations}")
    print(f"  Max Trials: {max_trials}")
    print(f"  Execution per Trial: {execution_per_trial}")
    print(f"  Epochs: {epochs}")
    
    print("\nMODEL PARAMETERS:")
    print(f"  Cycles: {cycle_list}")
    print(f"  Lead Times: {lead_time_list}")
    print(f"  Hours Back: {hours_back}")
    print(f"  Temperature Perturbations: {temperature_list}")
    print(f"  Output Units: {output_units}")
    print(f"  Units per Layer: {unit_list}")
    print(f"  Activations: {activation_list}")
    print(f"  Output Activation: {output_activation}")
    print(f"  Learning Rates: {learning_rate_list}")
    print(f"  Optimizers: {optimizer_list}")
    print(f"  Kernel Regularizer: {kernel_regularizer}")
    print(f"  Objective: {obj}")
    print(f"  Callback Monitor: {call_back_monitor}")
    print("=" * 60)

# Example usage
if __name__ == "__main__":
    # Setup global configuration variables
    setup_global_configuration()
    
    # Print configuration for verification
    print_configuration()
    