import numpy as np

def cobb_douglas_utility(inputs, parameters):
    """
    Calcula a utilidade Cobb-Douglas para o caso n-dimensional.

    Args:
        inputs (list): Lista de valores de entrada.
        parameters (list): Lista de parâmetros.

    Returns:
        float: Valor da utilidade Cobb-Douglas.
    """
    inputs = np.array(inputs)
    parameters = np.array(parameters)

    if len(inputs) != len(parameters):
        raise ValueError("O número de inputs deve ser igual ao número de parâmetros.")

    return np.prod(inputs ** parameters)


def constant_substitution_elasticity_utility(inputs, parameters, sigma):
    """
    Calcula a utilidade de elasticidade de substituição constante para o caso n-dimensional.

    Args:
        inputs (list): Lista de valores de entrada.
        parameters (list): Lista de parâmetros.
        sigma (float): Valor da elasticidade de substituição constante.

    Returns:
        float: Valor da utilidade de elasticidade de substituição constante.
    """
    inputs = np.array(inputs)
    parameters = np.array(parameters)

    if len(inputs) != len(parameters):
        raise ValueError("O número de inputs deve ser igual ao número de parâmetros.")

    return np.sum((inputs ** (parameters / sigma))) ** (sigma / (sigma - 1))


# Dados de exemplo
items = [2, 3, 4]
consumption_profiles = [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]]

# Cálculo da utilidade Cobb-Douglas
parameters_cd = [0.4, 0.6, 0.8]
cobb_douglas_utilities = []
for profile in consumption_profiles:
    utility = cobb_douglas_utility(profile, parameters_cd)
    cobb_douglas_utilities.append(utility)

# Cálculo da utilidade de elasticidade de substituição constante
parameters_esc = [0.3, 0.5, 0.7]
sigma = 0.9
constant_substitution_elasticity_utilities = []
for profile in consumption_profiles:
    utility = constant_substitution_elasticity_utility(profile, parameters_esc, sigma)
    constant_substitution_elasticity_utilities.append(utility)

print("Utilidades Cobb-Douglas:", cobb_douglas_utilities)
print("Utilidades de elasticidade de substituição constante:", constant_substitution_elasticity_utilities)
