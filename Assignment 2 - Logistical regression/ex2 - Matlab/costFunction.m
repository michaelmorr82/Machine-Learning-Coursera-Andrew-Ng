function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta,1);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta

    
hyp = sigmoid(X * theta); %this calcualtes g(theta0 * X0 + theta1 * X1 + theta2 * X2)
J = 1/m * sum (-y .* log(hyp) - (1 - y) .* log(1 - hyp));

for j = 1:1:n
    grad(j) = 1/m * sum((hyp - y) .* X(:,j));
end






% =============================================================

end
