function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

figure; % open a new figure window

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the training data into a figure using the 
%               "figure" and "plot" commands. Set the axes labels using
%               the "xlabel" and "ylabel" commands. Assume the 
%               population and revenue data have been passed in
%               as the x and y arguments of this function.
%
% Hint: You can use the 'rx' option with plot to have the markers
%       appear as red crosses. Furthermore, you can make the
%       markers larger by using plot(..., 'rx', 'MarkerSize', 10);


   %figure('Color',[1 1 1],...
   % 'Name',' Intensity Profile')
   % axes1 = axes(...
  %      'Color',[0.99 0.99 0.99],...
  %      'FontSize',20);
    plot(x, y,'rx', 'MarkerSize',10);
    title('Profit in $10,000', 'FontSize', 25);
    xlabel('Population of city in 10,000s','FontSize',20)
    ylabel('Probability Density Function (s/m)','FontSize',20)

% ============================================================

end
