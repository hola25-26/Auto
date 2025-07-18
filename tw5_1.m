opts = detectImportOptions('Mall_Customers.csv.xlsx');
opts = setvartype(opts,{'Gender'},'string');
data = readtable('Mall_Customers.csv.xlsx',opts);
GenderNumeric = double(data.Gender == "Female");
X = [GenderNumeric,data.Age,data.Annual_Income,data.Spending_Score];
k = 5;
[idx,C] = kmeans(X,k);
figure;
gscatter(X(:,3),X(:,4),idx);
hold on;
plot(C(:,3),C(:,4),'kx','MarkerSize',8,'LineWidth',2);
xlabel('Annual Income');
ylabel('Spending Score');
title('K mean clustering');
new_data = [0,25,50,60];
result = knnsearch(C,new_data);
fprintf('Cluster -----> %d\n',result);
plot(new_data(3),new_data(4),'gs','MarkerSize',7,'MarkerFaceColor','g');
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 4','Centroid','New Customer');
grid on;