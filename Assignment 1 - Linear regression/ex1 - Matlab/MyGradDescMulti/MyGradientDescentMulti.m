
%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:3);
y = data(:, 3);
m = length(y);
n = size(X,2);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

for t = 1 : 1 : n
    
    mu(1,t) = mean(X(:,t));
    sigma(1,t) = std(X(:,t));
    
    X_norm(:,t) = (X(:,t) - mu(1,t))/sigma(1,t);
end


% Add intercept term to X
X_norm = [ones(m, 1) X];
n = n + 1;

%X = 1.... size... rooms... cost



%=========================================================================
%                  GRADIENT DESCENT ALGORITHM
%=========================================================================
Grad_iter = 50;
alpha = 0.000000000001;

theta = zeros(1,n);
J = zeros(Grad_iter,1);

for iter = 1:1:Grad_iter
    
    hyp = theta(iter,1) * X_norm(:,1) + theta(iter,2) * X_norm(:,2) + theta(iter,3) * X_norm(:,3) + theta(iter,4) * X_norm(:,4);
 
    % Cost Function using normal technique
    J(iter)  = (1/(2 * m)) * sum(( hyp - y).^2);


    for f = 1 : 1 : n % f=feature
        dJ(iter,f) = 1/m * sum((hyp - y) .* X_norm(:,f));
        theta(iter, f) = theta(iter, f) - alpha * dJ(iter,f);
    end
    
    %update theta

    theta(iter + 1,:) = theta(iter,:);
    if iter == Grad_iter
       theta_final = theta(iter,:);
       J_final = J(iter);
    end
    

end

fprintf('DATA\n');
fprintf('--------------\n');
fprintf('Data File: ex1data2.txt\n');
fprintf('Number of features (n): %d\n', m);
fprintf('Number of training examples (m): %d\n', n);
fprintf('Features:\n');
fprintf('\t (x_0) = 1 \n');
fprintf('\t (x_1) = Size of House (ft^2) \n');
fprintf('\t (x_2) = Number of Bedrooms \n');
fprintf('Target (y): Houe Price\n');


fprintf('\nNORMALISTION\n');
fprintf('--------------\n');
fprintf('Average: \n');
fprintf('\t Average (/mu_x_1) = %.3f \n',mu(1));
fprintf('\t Average (/mu_x_2) = %.3f \n',mu(2));
fprintf('\t Average (/mu_y) = %.3f \n',mu(3));
fprintf('Standard deriation:\n');
fprintf('\t stvd (/sigma_x_1) = %.3f \n',mu(1));
fprintf('\t stdv (/sigma_x_2) = %.3f \n',mu(2));
fprintf('\t stdv (/sigma_y) = %.3f \n',mu(3));


fprintf('\n GRADIENT DESCENT\n');
fprintf('--------------\n');
fprintf('Learning rate (/alpha): %f \n', alpha);
fprintf('Number of iterations: %d\n', Grad_iter);

fprintf('\n FINAL RESULTS');
fprintf('--------------\n');
fprintf('Fitting parameters:\n');
fprintf('\t /theta_0: %f\n', theta_final(1));
fprintf('\t /theta_1: %f\n',theta_final(1));
fprintf('\t /theta_2: %f\n',theta_final(1));
fprintf('FInal Cost Function: %f\n',J_final);
fprintf('q\n');
fprintf('q\n');
fprintf('q\n');
fprintf('q\n');

%Testing the normalisation




 figure('Color',[1 1 1],...
    'Name',' ')
    axes1 = axes(...
        'Color',[0.99 0.99 0.99],...
        'FontSize',16);
    plot(J, 'LineWidth', 2); hold on;
    title('Cost Function', 'FontSize', 25);
    xlabel('Number of Iterations','FontSize',16)
    ylabel('Cost Function J(\theta)','FontSize',16)
          box('on');
      grid('on');
      
 figure('Color',[1 1 1],...
    'Name',' ')
    axes1 = axes(...
        'Color',[0.99 0.99 0.99],...
        'FontSize',16); 
    subplot(2,2,1)
      plot(theta(:,1),'r','LineWidth', 2); hold on;
    ylabel('\theta_0','FontSize',16)
    xlabel('Number of Iterations','FontSize',16)
    xlim([0 Grad_iter])
          box('on');
      grid('on');
      
      subplot(2,2,2)
      plot(theta(:,2),'g','LineWidth', 2); hold on;
    ylabel('\theta_1','FontSize',16)
    xlabel('Number of Iterations','FontSize',16)
    xlim([0 Grad_iter])
          box('on');
      grid('on');
      
      subplot(2,2,3)
      plot(theta(:,3),'b','LineWidth', 2); hold on;
      xlim([0 Grad_iter])
    ylabel('\theta_2','FontSize',16)
    xlabel('Number of Iterations','FontSize',16)
          box('on');
      grid('on');
      
       subplot(2,2,4)
      plot(theta(:,4),'k','LineWidth', 2); hold on;
      xlabel('Number of Iterations','FontSize',16)
    ylabel('\theta_3','FontSize',16)
    xlim([0 Grad_iter])
      box('on');
      grid('on');



