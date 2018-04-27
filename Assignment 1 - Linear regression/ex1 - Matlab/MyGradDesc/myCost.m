function J = myCost(x_data, y_data, theta)

m = length(y_data);

hyp = theta(:,1) * x_data(:,1) + theta(:,2) * x_data(:,2);

J = 1/(2 * m) * sum( hyp - y_data(:).^2); 
