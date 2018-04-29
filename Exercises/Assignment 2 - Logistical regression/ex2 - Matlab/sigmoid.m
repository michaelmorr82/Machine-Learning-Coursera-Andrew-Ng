function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

[m,n] = size(z);

for k = 1:1:m
    for  l = 1 : 1 : n 
        g(k,l) = 1/(1 + exp(-z(k,l)));
    end
    
end

% =============================================================

end
