
import argparse


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='AI Model Maker', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--enviroment',                         type=str,           default='schooner',                 help="Sets where we are running this on to set the std.out and std.err saving files.")
    parser.add_argument('--nogo',                               action='store_true',                                    help='Do not perform the experiment')
    parser.add_argument('--verbose', '-v',                      action='count',     default=0,                          help="Verbosity level")
    parser.add_argument('--model_type',                         type=str,           default='mlp',                      help="Model Architecture type")
    parser.add_argument('--results_folder',                     type=str,           default='results',                  help="Folder path to which to save the results to.")
    parser.add_argument('--experiment_number', '--en',          type=int,           default=0,                          help="Experiment number for running arrays through slurm.")

    parser.add_argument('--parallel_computing',                 action='store_true',                                    help="Use parallel computing technique")
    parser.add_argument('--pool',                               type=int,           default=7,                          help="How many processes to run in parallel in the multiprocessing pool")

    parser.add_argument('--jobid',                              type=int,           default=None,                       help="Slurms Job Id; for debugging")

    # Naming / Experiment Parameters

    parser.add_argument('--cycle',                   nargs='+', type=int,           default=[0],                        help="Cycle of data split for loading our dataset")
    parser.add_argument('--leadtime',                nargs='+', type=int,           default=[12],                       help="Leadtime we are predicting")
    parser.add_argument('--repetitions',                        type=int,           default=1,                          help="Number of times to repeat this experiment")

    parser.add_argument('--c_cycle',                            type=int,           default=0,                          help="current experiments: Leadtime we are predicting")
    parser.add_argument('--c_leadtime',                         type=int,           default=12,                         help="current experiments: Leadtime we are predicting")
    parser.add_argument('--c_repetitions',                      type=int,           default=1,                          help="current experiments: Number of times to repeat this experiment")

    # Training Parameters
    parser.add_argument('--lrate',                              type=float,         default=0.0001,                     help="Learning rate")
    parser.add_argument('--epochs',                             type=int,           default=100,                        help="Number of training epochs")
    parser.add_argument('--metrics',                 nargs='+', type=str,           default='mse',                      help="Metrics to evaluate the Model with.")

    # Data Parameters
    parser.add_argument('--data_set',                           type=str,           default='whole',                    help="Which dataset should get loaded")
    #parser.add_argument('',type=,help=)

    '''add these to a sub_parser for mlp'''
    # MLP (water temp) Parameters
    # parser.add_argument('--input_hours_forecast', type=int, default=24, help="Leadtime to forecast/predict")
    parser.add_argument('--atp_hours_back',                     type=int,           default=12,                         help="Hour to go back for Air Tempurature.")
    parser.add_argument('--wtp_hours_back',                     type=int,           default=12,                         help="Hour to go back for Water Tempurature.")

    # Network Parameters
    parser.add_argument('--dropout_rate',                       type=float,         default=None,                       help="dropout_rate rate")
    parser.add_argument('--spatial_dropout',                    type=float,         default=None,                       help="Spatial dropout rate")
    parser.add_argument('--l2', '--lambda_l2',                  type=float,         default=None,                       help="L2 Regularization")
    parser.add_argument('--n_hidden',                nargs='+', type=int,           default=None,                       help="Dense layer sizes")
    parser.add_argument('-l', '--loss_function',                type=str,           default='categorical_crossentropy', help="Loss Function")
    parser.add_argument('--activation_function',                type=str,           default='elu',                      help="Activation Function")
    parser.add_argument('--batch_size',                         type=int,           default=32,                         help="Batch Size")
    parser.add_argument('--patience',                           type=int,           default=25,                         help="Patience for Early Stoping")
    parser.add_argument('--min_delta',                          type=float,         default=0.01,                       help="Patience for Early Stoping")

    parser.add_argument('--lr_reducer_patience',                type=int,           default=15,                         help="Sets the patience parameter for the learning rate reducer callback")
    parser.add_argument('--early_stop_patience',                type=int,           default=45,                         help="Sets the patience parameter for the early stopping callback")

    # Input Layer Parameters # removed bc input layer does not recieve a number of nuerons, only recieves an input shape 
    # parser.add_argument('--input_neurons',                      type=int,           default=None,                       help="Number of Nuerons in Input Layer.")


    '''add these to a sub_parser for cnn'''
    parser.add_argument('--n_filters',               nargs='+', type=int,           default=None,                       help="Number of Conv filters")
    parser.add_argument('--kernel_sizes',            nargs='+', type=int,           default=None,                       help="Kernel sizes")
    parser.add_argument('--pooling',                 nargs='+', type=int,           default=None,                       help="Pooling sizes")

    parser.add_argument('--num_output_neurons',                 type=int,           default=1,                          help="Number of Nuerons in Output Layer.")

    # Modify Loss for experiments 4/1/2024
    # parser.add_argument('--modify_loss', type=bool, default=False, help="Number of Nuerons in Output Layer.")

    # dec 29th 2024 airplane to orlando edits
    parser.add_argument('--modify_sigma_loss',                  action='store_true',                                    help="Whether to use a sigma modification in loss func or not.")
    parser.add_argument('--sigma_threshold',                    type=float,         default=2,                       help="Threshold for sigma modification to loss func.")
    parser.add_argument('--sigma_regularization_parameter',     type=float,         default=0.1,                       help="Regularization Parameter for sigma loss func modification")

    parser.add_argument('--modify_mu_loss',                     action='store_true',                                    help="Whether to use a mu modification in loss func or not.")
    parser.add_argument('--mu_threshold',                       type=float,         default=0.5,                       help="Threshold for mu modification to loss func.")
    parser.add_argument('--mu_regularization_parameter',        type=float,         default=0.2,                       help="Regularization Parameter for mu loss func modification")


    return parser