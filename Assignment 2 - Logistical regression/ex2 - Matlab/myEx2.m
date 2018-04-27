%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('ex2data1.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

pos = find(y == 1);
neg = find(y == 0);


 figure('Color',[1 1 1],...
    'Name',' ')
    axes1 = axes(...
        'Color',[0.99 0.99 0.99],...
        'FontSize',20); 
      plot(X(pos,1), X(pos,2),'k+', 'LineWidth',2,'MarkerSize',10); hold on;
      plot(X(neg,1), X(neg,2),'ko', 'MarkerFaceColor','y','LineWidth',2,'MarkerSize',10); hold on;
      box('on');
      grid('on');
    title('Plotting Raw and Fitted Data', 'FontSize', 25);
    xlabel('Exam 1 Score','FontSize',20)
    ylabel('Exam 2 Score','FontSize',20)
    legend('Admitted','Not Admitted')
    
    
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters



J = 0;

iter = 100;
alpha = 0.0001;

theta = zeros(n + 1, iter);
dJ = zeros(n+1,1);

for k = 1:1:iter
    
    hyp = 1./(1 + exp(-X * theta(:,k))); %this calcualtes g(theta0 * X0 + theta1 * X1 + theta2 * X2)
    
    J(k) = 1/m * sum (-y .* log(hyp) - (1 - y) .* log(1 - hyp));

    for j = 1:1:n+1
        dJ(j) = 1/m * sum((hyp - y) .* X(:,j));
        theta(j,k + 1) = theta(j,k) - alpha * dJ(j);
    end   
    
    if k == iter
        J_final = J(k);
        theta_final = theta(:,k);
    end
end

J

 figure('Color',[1 1 1],...
    'Name',' ')
    axes1 = axes(...
        'Color',[0.99 0.99 0.99],...
        'FontSize',20); 
      plot(J,'LineWidth',2); hold on;
      box('on');
      grid('on');
    title('J', 'FontSize', 25);
    xlabel('interation','FontSize',20)
    ylabel('J(theta)','FontSize',20)
    legend('Admitted','Not Admitted')
    
    