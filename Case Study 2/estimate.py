def estimate_rf_training_times(base_time, base_params, param_grid):
    """
    Estimate the training time for different Random Forest configurations.

    :param base_time: The known training time for the base parameters.
    :param base_params: A dict of the base parameters (n_estimators and max_depth).
    :param param_grid: A dict of parameters to estimate training times for.
    :return: A dict with parameter combinations as keys and estimated times as values.
    """
    estimated_times = {}

    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            # Calculate time proportionally based on n_estimators and max_depth
            time_multiplier = (n_estimators / base_params['n_estimators']) * \
                              (max_depth / base_params['max_depth'])
            estimated_time = base_time * time_multiplier

            # Store the estimated time with the parameter combination
            params_tuple = (n_estimators, max_depth)
            estimated_times[params_tuple] = estimated_time

    return estimated_times

# Base training time and parameters
base_training_time = 291.5889766216278  # Time in seconds
base_parameters = {'n_estimators': 100, 'max_depth': 10}

# Parameter grid to estimate times for
rf_param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [10, 50, 100]
}

# Estimate training times
estimated_training_times = estimate_rf_training_times(base_training_time, base_parameters, rf_param_grid)

# Print estimated times
for params, time in estimated_training_times.items():
    print(f"Params (n_estimators={params[0]}, max_depth={params[1]}): Estimated Time = {time} seconds")
