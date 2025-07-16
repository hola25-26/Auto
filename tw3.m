data = readmatrix('swedish_insurance.csv.xlsx');
x = data(:,1);
y = data(:,2);
model = fitlm(x,y);
disp(model);
x_new = 6;
y_pred = predict(model,x_new);
fprintf('X = %.2f then Y = %.2f',x_new,y_pred);
figure;
plot(model);
hold on
plot(x_new,y_pred,'gs','MarkerSize',7,'MarkerFaceColor','m');
xlabel('Features');
ylabel('Targets');
title('Linear Regression');
legend('Training Data','Regression Line','Alternet Fits','New Data');
grid on;