function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

n = length(X(1,:));

for iter = 1:1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    % calculate derivatives

    
    h = (theta' * X')'
    
    
    % hypotesis for multivariant gradient descent 
    
    for k = 1 : 1 : n
        %dJ_theta = (1/m) * sum((h - y).* X (:,k));
        dJ_theta(k) = (1/m) * sum((h - y) .* X (:,k));
    end
    
%     %updating thetas
     for l = 1 : 1 : n
        temp(l) = theta(l) - alpha * dJ_theta(l);
        theta(l) = temp(l); 
     end
%        
    % ============================================================

    % Save the cost J in every iteration    
 J_history = computeCostMulti(X, y, theta);

end

end
