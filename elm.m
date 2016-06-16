data = csvread('train.csv',1,0)
m = size(data,1);
X = data(:,1:size(data,2));
y = data(:,0);
m = size(y,1);
y_oneshot = zeros(m, 10);

for i = 1:m,
   digit = int(y(i));
   y_oneshot(i, digit) = 1.0;
   
end;

t = y_oneshot;


nh = 100;
d = 784;

Xa = [ones(p,1),X];                     % append ones to X (bias term)
Wh = rand(d+1,nh);                      % hidden weights from U(0,1) 
Wh(2:end,:)= 2*Wh(2:end,:)-1;           % remap nonbias weights to -1,+1
H = 1./(1+exp(-Xa*Wh));                 % design matrix 
%Hi = pinv(H);                          % calculate the pseudoinverse of the design matrix
Hi=pinv(H'*H+lamda*eye(size(H'*H)))*H'; % regularised pseudoinverse
w = Hi*t;                               % fitted linear weights 


tpred = H* w;
HitAccuracy = 0;

 for i = 1 : m,
        [x, label_index_actual]=max(tpred(i,:));
        [x, label_index_expected]=max(t(i,:));
        if label_index_actual==label_index_expected
            HitAccuracy = HitAccuracy + 1;
        end
  end
percent = (HitAccuracy/m)*100




%n = size(Q,1);                          % n = number of query points  
%Qa = [ones(n,1),Q];                     % append ones to Q (bias term)
%Ht = 1./(1+exp(-Qa*Wh));                % outputs of basis functions at query points
%ypred = Ht*w;                           % predicted targets for query points Q 
%r = t - H*w ;                           % internal residuals (fit errors on training points)

%rse = sum(r.^2)/p;                      % squares sum fit error(fit errors on training points)
%H = H*Hi;                               % calculate the hat matrix X(X'X)^-1)X'
%hd = diag(H);                           % extract the diagonal elements
%prse = sum((r./(1-hd)).^2)/p;
