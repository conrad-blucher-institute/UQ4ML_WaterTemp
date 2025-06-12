# add documentation here -hector 6-11

if __name__ == "__main__":

    # Parse incoming command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    if args.model == 'pnn':
        # call cmd_ai_builder.py
        pass

    elif args.model == 'mse':
        # call mme_runner.py
        pass

    elif args.model == 'crps':
        # call mme_runner.py
        pass