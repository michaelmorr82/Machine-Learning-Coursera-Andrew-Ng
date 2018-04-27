% this file should calculate the values of theta0 and theta1 for the data
% given in  ex1data1.txt using the gradient descent method

clear all 
clc
for i = 1:1:10
    delete(figure(i))
end


%=========================================================================
%                  IMPORT DATA
%=========================================================================
data_file = 'ex1data1.txt';
data = load(data_file);

x_data = (data(:,1));
y_data = data(:,2);

m = length(data); % number of samples

x_data = [ones(m,1) x_data]; % put list on ones at begining on x_data; this is for X_0

theta = [0, 0]; % intialise first theta0 = theta1 = 0

%plot raw data
 figure('Color',[1 1 1],...
    'Name',' Intensity Profile')
    axes1 = axes(...
        'Color',[0.99 0.99 0.99],...
        'FontSize',20); 
      plot(x_data(:,2), y_data,'rx', 'MarkerSize',10); box('on');grid('on');
    title('Plotting Raw Data', 'FontSize', 25);
    xlabel('X - Data','FontSize',20)
    ylabel('Y - Data','FontSize',20)
    ylim([-5 25])
    xlim([0 25])

%=========================================================================
%                  INITIAL COST FUCNTION
%=========================================================================

%initial cost function for thete0 = theta1 = 0 
J = 1/(2 * m) * sum((theta(:,1) * x_data(:,1) + theta(:,2) * x_data(:,2) - y_data(:)).^2); 

J = myCost(x_data, y_data, theta); % turns the above equation into a function call, again tests for initial theta

% print the initial results
fprintf('INITIAL VALUES\n')
fprintf('data file: %s\n',data_file);
fprintf('Initial Theta_0 = %.2f, theta1 = %.2f \n', theta(1), theta(2)); 

fprintf('Initial Cost Function: J(theta_0, theta_1) = %.2f \n', J); 

%=========================================================================
%                  GRADIENT DESCENT ALGORITHM
%=========================================================================
grad_desc_i = 1500;
alpha = 0.01;



for i = 1:1:grad_desc_i
    %calculate cost function using values of theta
    J_grad_desc(i)  = myCost(x_data, y_data, [theta(1), theta(2)]);
    
    %make list of theta guesses
    theta_guess(i,:) = [theta(1), theta(2)];
    
    h_theta = theta(1) .* x_data(:,1) + theta(2) .* x_data(:,2);
    
    % calculate derivative part
    dJ_theta_0 = (1/m) * sum(h_theta - y_data(:));
    dJ_theta_1 = (1/m) * sum((h_theta - y_data(:)).* x_data(:,2));
    dJ_theta_0
    
    %calculate new value of thetas
    temp_theta0 = theta(1) - (1/m) * dJ_theta_0 ;
    temp_theta1 = theta(2) - alpha * dJ_theta_1 ;
    
   
    if i == grad_desc_i
        J_grad_final = J_grad_desc(i);
    end
    
     
     % update thetas
    theta(1) = temp_theta0;
    theta(2) = temp_theta1;

end

y_data_fit = x_data * theta'; 

alpha_txt = ['\alpha = ' num2str(alpha)];
theta0_txt = ['\theta_0 = ' num2str(theta(1))];
theta1_txt = ['\theta_1 = ' num2str(theta(2))];
J_txt = ['J(\theta_0, \theta_1) = ' num2str(J_grad_final)]


fprintf('\n\nRESULTS AFTER GRADIENT DESCENT\n');
fprintf('Final Results: \n\tCost Function: %.2f \n\ttheta_0 = %.2f\n\ttheta_1 = %.2f\n',J_grad_final, theta(1),theta(2));

%=========================================================================
%                         3-D Data
%=========================================================================
theta0_vals = -1000:1:1000;
theta1_vals = -100:1:100;

% initialize J_vals to a matrix of 0's
%J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  J_vals(i,j) = myCost(x_data, y_data, [theta0_vals(i), theta1_vals(j)]);
    end
end

%=========================================================================
%                  PLOTTING RESULTS
%=========================================================================

 figure('Color',[1 1 1],...
    'Name',' ')
    axes1 = axes(...
        'Color',[0.99 0.99 0.99],...
        'FontSize',20); 
      plot(x_data(:,2), y_data,'rx', 'MarkerSize',10); hold on;
      plot(x_data(:,2), y_data_fit,'LineWidth', 2);
      box('on');
      grid('on');
    title('Plotting Raw and Fitted Data', 'FontSize', 25);
    xlabel('X - Data','FontSize',20)
    ylabel('Y - Data','FontSize',20)
    ylim([-5 25])
    xlim([0 25]) 
         annotation1 = annotation(...
         'textbox',...
         [0.6 0.26 0.27 0.11],...
         'LineStyle','none',...
         'Color',[1 0 0],...
         'FitHeightToText','on',...
         'FitBoxToText','on',...
         'FontWeight','bold',...
         'Fontsize', 18,...
         'String',{alpha_txt, theta0_txt, theta1_txt, J_txt});

 figure('Color',[1 1 1],...
    'Name',' ')
    axes1 = axes(...
        'Color',[0.99 0.99 0.99],...
        'FontSize',20); 
      plot(J_grad_desc, 'LineWidth', 2);
      box('on');
      grid('on');
    title('Cost Function J(\theta_0 \theta_1)', 'FontSize', 20);
    xlabel('Gradient Descent Iteration','FontSize',20)
    ylabel('Cost Function J(\theta_0, \theta_1)','FontSize',20)
    
 figure('Color',[1 1 1],...
    'Name',' ')
    axes1 = axes(...
        'Color',[0.99 0.99 0.99],...
        'FontSize',20); 
      plot(theta_guess(:,1),'r', 'LineWidth', 2);hold on;
      plot(theta_guess(:,2), 'b','LineWidth', 2);
      box('on');
      grid('on');
    title('Variation in values of \theta''s', 'FontSize', 20);
    xlabel('Gradient Descent Iteration','FontSize',20)
    ylabel('Values of \theta''s','FontSize',20)
    legend('\theta_0','\theta_1')


 figure('Color',[1 1 1],...
    'Name',' ')
    axes1 = axes(...
        'Color',[0.99 0.99 0.99],...
        'FontSize',20);
surf(theta0_vals, theta1_vals, J_vals');shading interp; colorbar
title('Variation in Cost Function', 'FontSize', 20);
xlabel('Values of \theta_0'); 
ylabel('Values of \theta_1');
zlabel('Cost Function J(\theta_0, \theta_1)');


