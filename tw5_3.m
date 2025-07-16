function DivisiveClustering(X,name,depth)
 if nargin<2
 name ='Root'; depth = 0;
 fprintf('Divisive Clustering\n');
 end
 if size(X,1)<=2
 fprintf('%s%s Leaf Clusters (%d samples)\n',repmat(' ',1,depth),name,size(X,1));
 return;
 end
 [idx,~] = kmeans(X,2);
 for i = 1:2
 subX = X(idx==i,:);
 subname = sprintf('%s-%d',name,i);
 fprintf('%s%s - %d samples\n',repmat(' ',1,depth),subname,size(subX,1));
 DivisiveClustering(subX,subname,depth+1);
 end
end
X = zscore(table2array(readtable('wine_dataset_for_hiearchical_clusterig.csv.xlsx')));
DivisiveClustering(X);
