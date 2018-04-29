function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


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








% =========================================================================



hold off;

end
