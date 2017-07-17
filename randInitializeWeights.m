function W = randInitializeWeights(L_in, L_out, eps)

W = rand(L_out, 1+L_in) * 2 * eps - eps;

end
