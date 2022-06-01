% What is this code about Paul?

% Put some date and location of which this code was created by who

clc, clear all, close all
import casadi.*
%load flapping wing simulation states and aerodynamic forces
load data.mat

%import casadi functions
func_dyn = external('f', 'func_aerobat_dyn.mexw64');

%extract flapping wing data
 x=data.x;  %states
 u=data.u;  % torque input
 ua=data.ua;    %aerodynamic forces
 N_sim=size(x,2);   %simulation length
 params.dt = 1e-4;  %sample rate
 
 %% neural  net parameters setup
 nh=2;%select number of neurons of the neural net
 Z=zeros(nh,1);% set recurrent weights

 
 M=nh;% number of hidden layers

 params.M=M;
 nx=size(x,1);%size of NN input
 ny=nx/2;%size of NN output
 params.nx=nx;
 params.ny=ny;

% Initialize Neural net parameters
 params.A = 1e-1*randn(M,nx);
 params.O=  1e-1*randn(M,nh);
 params.b = 1e-1*randn(M,1);
 params.C = 1e-1*randn(ny,M);
 params.d = zeros(ny,1); % initialize to mean of y
  
% transform neural net weights from matrix to vector
Wk=[reshape(params.A,[M*nx,1]);reshape(params.O,[M*nh,1]);reshape(params.b,[M,1])...
    ;reshape(params.C,[M*ny,1]);params.d];

%% initalize cubature kalman filter parameters
n=length(Wk);%length of neural net weights vector
params.n=n;
m=2*n;%set number of cubature points

%covariance matrices
P=1e-3*eye(n);
Q=1e-3*eye(n);
R=1e-3*eye(ny);


%% Aerobat parameters
p.g = 9.8; 

% length parameters (about zero configuration)
p.body_radius = 0.025;    % 2.5 cm
p.body_length = 0.1;      % 10cm
p.humerus_length = 0.05;  % 5cm
p.humerus_radius = 0.002;  % 2 mm
p.radius_length = 0.12;   % 13 cm
p.chord_length = 0.1;    % 16 cm
p.wing_thickness = 0.00025;    % 0.25 mm

p.L1_L = [0;0.0207;0.0075];   % Body CoM to shoulder
p.L2_L = [0;p.humerus_length;0];          % shoulder to elbow
p.L3_L = [0;p.radius_length;0];           % elbow to wingtip (midpoint)
p.L1_R = p.L1_L;           
p.L2_R = p.L2_L; 
p.L3_R = p.L3_L;

% mass parameters
p.mb = 0.010; 
p.ma = 0.001; 
p.mw = 0.004; 

% inertia parameters (diagonal components, x-y-z)
p.Ib_1 = p.mb/2*p.body_radius^2;
p.Ib_2 = p.mb/12*(3*p.body_radius^2 + p.body_length^2);
p.Ib_3 = p.mb/12*(3*p.body_radius^2 + p.body_length^2);
p.Ib = [p.Ib_1; p.Ib_2; p.Ib_3]; % cylinder

p.Ia_1 = p.ma/12*(3*p.humerus_radius^2 + p.humerus_length^2);
p.Ia_2 = p.ma/12*(3*p.humerus_radius^2 + p.humerus_length^2);
p.Ia_3 = p.ma/2*p.humerus_radius^2 ; % This inertia is too small
p.Ia = [p.Ia_1; p.Ia_2; p.Ia_3]; % cylinder

p.Iw_1 = p.mw/12*(p.radius_length^2 + p.wing_thickness^2);
p.Iw_2 = p.mw/12*(p.radius_length^2 + p.chord_length^2);
p.Iw_3 = p.mw/12*(p.chord_length^2 + p.wing_thickness^2); % This inertia is kind of small
p.Iw = [p.Iw_1; p.Iw_2; p.Iw_3]; % box


params_aero =  [p.mb; p.ma; p.mw; p.g; p.L1_L ;p.L2_L; p.L3_L; p.L1_R; p.L2_R; p.L3_R;...
            p.Ib ; p.Ia; p.Iw];


  

for i=1:N_sim 
    

   
 %retransform neural network wieights from vector form to matrix form
    params.A=reshape(Wk(1:M*nx),M,nx);
    params.O=reshape(Wk(M*nx+1:M*nx+M*nh),M,nh);
    params.b=reshape(Wk(M*nx+M*nh+1:M*nx+M*nh+M),M,1);
    params.C=reshape(Wk(M*nx+M*nh+M+1:M*nx+M*nh+M+M*ny),ny,M);
    params.d=reshape(Wk(M*nx+M*nh+M+M*ny+1:end),ny,1);
    [f,Z]=mlpModel(x(:,i),Z,params);%calculate neural net output
    
%update weigths each each 100 sample
 if (i==1 || rem(i-1,1)==0)
    
  %% cubature kalman algorithm
  cubature_points= cubaturepoints(P,Wk);%generate cubature points
  xstar=cubature_points;
  x_hat=mean(xstar,2);
  x_hat=full(x_hat);
  P=1/m*xstar*xstar' - x_hat*x_hat' + Q;%update covariance matrix 
  P=full(P);

   
  %evaluate cubature points
  Wk_c= cubaturepoints(P,x_hat);
 for j=1:m 
    params.A=reshape(Wk_c(1:M*nx,j),M,nx);
    params.O=reshape(Wk_c(M*nx+1:M*nx+M*nh,j),M,nh);
    params.b=reshape(Wk_c(M*nx+M*nh+1:M*nx+M*nh+M,j),M,1);
    params.C=reshape(Wk_c(M*nx+M*nh+M+1:M*nx+M*nh+M+M*ny,j),ny,M);
    params.d=reshape(Wk_c(M*nx+M*nh+M+M*ny+1:end,j),ny,1);
    [f1(:,j),~]=mlpModel(x(:,i),Z,params);%calcualate neural network output of each cubature point
 end 
z=f1;
  
  z_hat=mean(z,2);
  %update covariances
  Pzz=1/m*z*z' - z_hat*z_hat'+ R ; 
  Pxz=1/m*Wk_c*z' - x_hat*z_hat' ;
  %ucalculate kalman gain
  Kk=Pxz*inv(Pzz);
  %update weights wk+1
  Wk=x_hat+Kk*(ua(:,i)-z_hat);
  
  %update covariance matrix
  P= P-Kk*Pzz*Kk';
 
 end
%store resulting estimated weights and neural net output
  data.Wk(:,i)=Wk;
  data.f(:,i)=f;
  data.P(:,:,i)=P;
  
  if i>=2 
      deltaW=data.Wk(:,i)-data.Wk(:,i-1);
      deltaP=data.P(:,:,i)-data.P(:,:,i-1);
 
  
  end 
  
end 
    
    
% figure(1)
%  
%  for i=1:2:13
%  subplot(7,2,i),plot(data.t(1:end-1), data.ua(i,2:end),data.t(1:end-1), data.f(i,2:end))
%  title('AeroDynamic Forces')
%  legend('actual','estimated')
%  subplot(7,2,i+1),plot(data.t(1:end-1), data.ua(i+1,2:end),data.t(1:end-1), data.f(i+1,2:end))
%  title('AeroDynamic Forces')
%  legend('actual','estimated')
%  end 
   
%calculate rms error
rms=sqrt(mean((ua-data.f).^2,2));


%% plots 
xLim = [0,1];
yLim = [];
lineWidth =2;
fig_name = 'x, y, z Gen. Aero. Forces';
fh=figure('Name',fig_name);

% Plot Aerodynamic x, y, z component of aero force
subplot(3,1,1),plot(data.t(1:end-1), data.ua(1,2:end),data.t(1:end-1), data.f(1,2:end),'linewidth',lineWidth);
title('Gen. Aero. Forces in  x Direction');
legend('Actual Model','Estimated Model');
xlim(xLim);
xlabel('Time (sec)')
ylabel('Force (N)')

% closeup view
% xLim = [1,1.25];
% yLim = [100,250];
% ax_Pos = [0.5,0.8,0.1,0.1];
% ah = axes('Position',ax_Pos);
% box(ah,'on');
% plot(ah, data.t(1:end-1), data.ua(1,2:end),data.t(1:end-1), data.f(1,2:end));
% xlim(ah, xLim);
% ylim(ah, yLim);
% set(ah, 'Yticklabel',[]);
% 
% 
xLim = [0,1];
subplot(3,1,2),plot(data.t(1:end-1), data.ua(2,2:end),data.t(1:end-1), data.f(2,2:end),'linewidth',lineWidth);
title('Gen. Aero. Forces in y Direction');
legend('Actual Model','Estimated Model');
xlim(xLim);
xlabel('Time (sec)')
ylabel('Force (N)')

% closeup view
% xLim = [0,0.25];
% yLim = [-1,1];
% ax_Pos = [0.3,0.51,0.1,0.1];
% ah = axes('Position',ax_Pos);
% box(ah,'on');
% plot(ah,data.t(1:end-1), data.ua(2,2:end),data.t(1:end-1), data.f(2,2:end));
% xlim(ah, xLim);
% ylim(ah, yLim);
% set(ah, 'Yticklabel',[]);

xLim = [0,1];
subplot(3,1,3),plot(data.t(1:end-1), data.ua(3,2:end),data.t(1:end-1), data.f(3,2:end),'linewidth',lineWidth);
title('Gen. Aero. Forces in  z Direction');
legend('Actual Model','Estimated Model');
xlim(xLim);
xlabel('Time (sec)')
ylabel('Force (N)')

% closeup view
% xLim = [1,1.25];
% yLim = [-180,170];
% ax_Pos = [0.5,0.21,0.1,0.1];
% ah = axes('Position',ax_Pos);
% box(ah,'on');
% plot(ah,data.t(1:end-1), data.ua(3,2:end),data.t(1:end-1), data.f(3,2:end));
% xlim(ah, xLim);
% ylim(ah, yLim);
% set(ah, 'Yticklabel',[]);

print(figure(1), '-painters', '-dsvg', '-r600', ['figures\',fig_name,'.svg']);


%% Please plot Gen. Aero. Forces for Roll, Pitch and Yaw just like above
xLim = [0,1];
yLim = [];
fig_name = 'r, p, y Gen. Aero. Forces';
fh=figure('Name',fig_name);

subplot(3,1,1),plot(data.t(1:end-1), data.ua(4,2:end),data.t(1:end-1), data.f(4,2:end),'linewidth',lineWidth)
title('Gen. Aero. Forces on Roll angle ')
legend('Actual Model','Estimated Model')
xlim(xLim);
xlabel('Time (sec)')
ylabel('Force (N)')

% %close up
% xLim = [0,0.25];
% yLim = [-.5,.5];
% ax_Pos = [0.4,0.8,0.1,0.1];
% ah = axes('Position',ax_Pos);
% box(ah,'on');
% plot(ah,data.t(1:end-1), data.ua(4,2:end),data.t(1:end-1), data.f(4,2:end));
% xlim(ah, xLim);
% ylim(ah, yLim);
% set(ah, 'Yticklabel',[]);
 

xLim = [0,1];
yLim = [];
subplot(3,1,2),plot(data.t(1:end-1), data.ua(5,2:end),data.t(1:end-1), data.f(5,2:end),'linewidth',lineWidth)
title('Gen. Aero. Forces on Pitch angle ')
legend('Actual Model','Estimated Model')
xlim(xLim);
xlabel('Time (sec)')
ylabel('Force (N)')

% %close up
% xLim = [1,1.25];
% yLim = [-20,20];
% ax_Pos = [0.5,0.51,0.1,0.1];
% ah = axes('Position',ax_Pos);
% box(ah,'on');
% plot(ah,data.t(1:end-1), data.ua(5,2:end),data.t(1:end-1), data.f(5,2:end));
% xlim(ah, xLim);
% ylim(ah, yLim);
% set(ah, 'Yticklabel',[]);
 
xLim = [0,1];
yLim = [];
subplot(3,1,3),plot(data.t(1:end-1), data.ua(6,2:end),data.t(1:end-1), data.f(6,2:end),'linewidth',lineWidth)
title('Gen. Aero. Forces on Yaw angle ')
legend('Actual Model','Estimated Model')
xlim(xLim);
xlabel('Time (sec)')
ylabel('Force  (N)')

% %close up
% xLim = [0,0.2];
% yLim = [-.5,.5];
% ax_Pos = [0.4,0.21,0.1,0.1];
% ah = axes('Position',ax_Pos);
% box(ah,'on');
% plot(ah,data.t(1:end-1), data.ua(6,2:end),data.t(1:end-1), data.f(6,2:end));
% xlim(ah, xLim);
% ylim(ah, yLim);
% set(ah, 'Yticklabel',[]);
 
print(figure(2), '-painters', '-dsvg', '-r600', ['figures\',fig_name,'.svg']);

%% Plot plot Gen. Aero. Forces for 4 DoFs of Right Wing just like above

xLim = [0,1];
yLim = [];
fig_name = 'Left wing Gen. Aero. Forces';
fh=figure('Name',fig_name);

subplot(4,1,1),plot(data.t(1:end-1), data.ua(7,2:end),data.t(1:end-1), data.f(7,2:end),'linewidth',lineWidth)
title('Gen. Aero. Forces on Plunge')
legend('Actual Model','Estimated Model')
xlim(xLim);
xlabel('Time (sec)')
ylabel('Force (N)')
% 
% %close up
% xLim = [1,1.25];
% yLim = [-10,0];
% ax_Pos = [0.5,0.84,0.065,0.065];
% ah = axes('Position',ax_Pos);
% box(ah,'on');
% plot(ah,data.t(1:end-1), data.ua(7,2:end),data.t(1:end-1), data.f(7,2:end),'-b','-r');
% xlim(ah, xLim);
% ylim(ah, yLim);
% set(ah, 'Yticklabel',[]);
 

xLim = [0,1];
yLim = [];
subplot(4,1,2),plot(data.t(1:end-1), data.ua(8,2:end),data.t(1:end-1), data.f(8,2:end),'linewidth',lineWidth)
title('Gen. Aero. Forces on Mediolateral')
legend('Actual Model','Estimated Model')
xlim(xLim);
xlabel('Time (sec)')
ylabel('Force (N)')

% %close up
% xLim = [0.5,0.75];
% yLim = [-15,0];
% ax_Pos = [0.38,0.62,0.065,0.065];
% ah = axes('Position',ax_Pos);
% box(ah,'on');
% plot(ah,data.t(1:end-1), data.ua(8,2:end),data.t(1:end-1), data.f(8,2:end));
% xlim(ah, xLim);
% ylim(ah, yLim);
% set(ah, 'Yticklabel',[]);
 
xLim = [0,1];
yLim = [];
subplot(4,1,3),plot(data.t(1:end-1), data.ua(9,2:end),data.t(1:end-1), data.f(9,2:end),'linewidth',lineWidth)
title('Gen. Aero. Forces on Elbow')
legend('Actual Model','Estimated Model')
xlim(xLim);
xlabel('Time (sec)')
ylabel('Force (N)')

% %close up
% xLim = [1.5,1.75];
% yLim = [-5,0];
% ax_Pos = [0.6,0.4,0.065,0.065];
% ah = axes('Position',ax_Pos);
% box(ah,'on');
% plot(ah,data.t(1:end-1), data.ua(9,2:end),data.t(1:end-1), data.f(9,2:end));
% xlim(ah, xLim);
% ylim(ah, yLim);
% set(ah, 'Yticklabel',[]);
 

xLim = [0,1];
yLim = [];
subplot(4,1,4),plot(data.t(1:end-1), data.ua(10,2:end),data.t(1:end-1), data.f(10,2:end),'linewidth',lineWidth)
title('Gen. Aero. Forces on Feathering')
legend('Actual Model','Estimated Model')
xlim(xLim);
xlabel('Time (sec)')
ylabel('Force (N)')

% %close up
% xLim = [0.5,0.75];
% yLim = [-2,0];
% ax_Pos = [0.4,0.18,0.065,0.065];
% ah = axes('Position',ax_Pos);
% box(ah,'on');
% plot(ah,data.t(1:end-1), data.ua(10,2:end),data.t(1:end-1), data.f(10,2:end));
% xlim(ah, xLim);
% ylim(ah, yLim);
% set(ah, 'Yticklabel',[]);



print(figure(3), '-painters', '-dsvg', '-r600', ['figures\',fig_name,'.svg']);
% Plot plot Gen. Aero. Forces for 4 DoFs of Left Wing just like above


%  print(joint_forces, '-painters', '-dpng', '-r600', 'estimated aero forces on rpy.png');

function xstar= F(cubature_points, params,x,u,func_dyn,params_aero)
n=params.n;%dimension of the process
m=2*n;%cubature number
nx=params.nx;%input of NN
ny=params.ny;%output of NN
Mn=params.M;%number of hidden layers
%extract states from the process
Wk=cubature_points(1:n-nx,:);
xk_1=cubature_points(n-nx+1:n-nx+nx/2,:);
xk_2=cubature_points(n-nx+nx/2+1:end,:);

%get aerobat matrices
[M,Cdq,G] = func_dyn(x(1:14),x(15:28),params_aero) ;
 M=full(M);
 h=Cdq+G;
 %% evaluate cubature points
for j=1:m
    params.A=reshape(Wk(1:Mn*nx,j),Mn,nx);
    params.b=reshape(Wk(Mn*nx+1:Mn*nx+Mn,j),Mn,1);
    params.C=reshape(Wk(Mn*nx+Mn+1:Mn*nx+Mn+Mn*ny,j),ny,Mn);
    params.d=reshape(Wk(Mn*nx+Mn+Mn*ny+1:end,j),ny,1);
    f(:,j)=mlpModel([xk_1(:,j);xk_2(:,j)],params);
end 
g=params.dt*inv(M)*(u+f-h) +xk_2;
%calculate process
Wk=eye(n-nx)*cubature_points(1:n-nx,:);
xk_1=params.dt*xk_2 +xk_1;
xk_2=g;

xstar=[Wk;xk_1;xk_2];


end 

function cubature_points = cubaturepoints(P,xk )

L=chol(P,'lower');%generate cholewsky matrix
d=length(xk);
m=2*d;
num=0.05*sqrt(m/2);

xi=[num*eye(d),-num*eye(d)];

cubature_points= xk + L*xi;%get cubature points


end 



%neural network function
function [H,Z] = mlpModel(X,Z,params)
N = size(X,2);% number of samples
nY = length(params.d);                  % number of outputs
U = params.A*X + 0*params.O*Z+ repmat(params.b,1,N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
Z = activationFunction(U);              % z \in R^nP, using nP instead of nPerceptons
V = params.C*Z + repmat(params.d,1,N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
H = V; % linear output layer activations
%H = exp(V)./repmat(sum(exp(V),1),nY,1); % softmax nonlinearity for second/last layer
% Activate the softmax function to make this MLP a model for class posteriors
%
end 

function out = activationFunction(in)
% Pick a shared nonlinearity for all perceptrons: sigmoid or ramp style...
% You can mix and match nonlinearities in the model.
% However, typically this is not done; identical nonlinearity functions
% are better suited for parallelization of the implementation.
%out = 1./(1+exp(-in)); % Logistic function - sigmoid style nonlinearity
%out = in./sqrt(1+in.^2); % ISRU - ramp style nonlinearity
out=log(1+exp(in));
%out=tanh(in);
%out=max(0,in);

end 


