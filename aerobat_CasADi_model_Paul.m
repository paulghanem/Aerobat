
 %% Aerobat symbolic model using CasADi
%
% 4-7-2021
% Author: Paul Ghanem
% Northeastern University, Boston, USA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all, close all, clc

tic 
import casadi.*


mb=MX.sym('mb'); 
ma=MX.sym('ma'); 
mw=MX.sym('mw'); 
g=MX.sym('g'); 


% Length vector variables (constant)
L1_L = MX.sym('L1_L', 3,1);   % body to shoulder (left)
L2_L = MX.sym('L2_L', 3,1);   % shoulder to elbow (left)
L3_L = MX.sym('L3_L', 3,1 );   % elbow to wingtip (left)
L1_R = MX.sym('L1_R', 3,1);   % body to shoulder (right)
L2_R = MX.sym('L2_R', 3,1);   % shoulder to elbow (right)
L3_R = MX.sym('L3_R', 3,1);   % elbow to wingtip (right) 

% Aerodynamics length vector variables (elbow to aero force location)
La_L = MX.sym('La_L', 3,1 );
La_R = MX.sym('La_R', 3,1);

% Inertia variables (constant)
Ib = MX.sym('Ib', 3,1); 
Ia = MX.sym('Ia', 3,1); 
Iw = MX.sym('Iw', 3,1); 
I_body = diag (Ib);    % body inertia matrix
I_link = diag (Ia);    % link inertia matrix
I_plate = diag (Iw);   % plate inertia matrix

% time varying variables declarations -------------------------------------

% Body degrees of freedom
px=MX.sym('px');
py=MX.sym('py');
pz=MX.sym('pz');

qx=MX.sym('qx');
qy=MX.sym('qy');
qz=MX.sym('qz');

qp_L=MX.sym('qp_L');
qm_L=MX.sym('qm_L');
qe_L=MX.sym('qe_L');
qf_L=MX.sym('qf_L');

qp_R=MX.sym('qp_R');
qm_R=MX.sym('qm_R');
qe_R=MX.sym('qe_R');
qf_R=MX.sym('qf_R');

pos_body = [px;py;pz];    % body CoM linear position
angles = [qx;qy;qz];    % body angles

Rx=@(th) [ 1 0 0 ; 
        0 cos(th) -sin(th) ; 
        0 sin(th) cos(th)]; 
Ry=@(th)[cos(th) 0 sin(th) ; 
         0   1  0 ; 
         -sin(th) 0 cos(th)]; 
     
Rz=@(th)[cos(th) -sin(th) 0 ; 
        sin(th) cos(th) 0 ;
        0  0  1]; 
        
R_body=Rz(qz)*Ry(qy)*Rx(qx); 



pxD=MX.sym('pxD');
pyD=MX.sym('pyD');
pzD=MX.sym('pzD');

qxD=MX.sym('qxD');
qyD=MX.sym('qyD');
qzD=MX.sym('qzD');

qpD_L=MX.sym('qpD_L');
qmD_L=MX.sym('qmD_L');
qeD_L=MX.sym('qeD_L');
qfD_L=MX.sym('qfD_L');

qpD_R=MX.sym('qpD_R');
qmD_R=MX.sym('qmD_R');
qeD_R=MX.sym('qeD_R');
qfD_R=MX.sym('qfD_R');

% Arm-wing degrees of freedom
% Order: plunge, mediolateral, elbow extension, feathering
theta_L = [qp_L;qm_L;qe_L;qf_L];    % angles L 
theta_R = [qp_R;qm_R;qe_R;qf_R];      % angles R
thetaD_L = [qpD_L;qmD_L;qeD_L;qfD_L];   % angles rate of change L
thetaD_R = [qpD_R;qmD_R;qeD_R;qfD_R];   % angles rate of change R

vel_body = [pxD;pyD;pzD];   % body CoM linear velocity
anglesD = [qxD;qyD;qzD];   % body angular rates in inertial frame


% State variables
q = [pos_body; angles; ...
      theta_L; theta_R; ];
  
qd = [vel_body; anglesD; ...
      thetaD_L;  thetaD_R; ];
  
  
  %% Kinematics Formulation

% Rotation matrices
% Defined about the previous frame of reference in the kinematic chain.
% order: humerus = [plunge, medio], radius = [ extension,feather]
Reflection_Y=[1 0 0; 0 -1 0;0 0 1];
Rot1_L = Rz(-qm_L)*Rx(qp_L); % plunge then medio
Rot2_L = Rx(qe_L)*Ry(-qf_L); % feather then elbow extension
Rot1_R = Rz(qm_R)*Rx(-qp_R);
Rot2_R = Rx(-qe_R)*Ry(-qf_R);

% Angular velocities (about their base frame)
w1_L = [0;0;-qmD_L] + Rz(-qm_L)*[qpD_L;0;0];
w2_L = [qeD_L;0;0] + Rx(qe_L)*[0;-qfD_L;0];

w1_R = [0;0;qmD_R] + Rz(qm_R)*[-qpD_R;0;0];
w2_R = [-qeD_R;0;0] + Rx(-qe_R)*[0;-qfD_R;0];

% Linear positions
com_link_L = pos_body + R_body * (L1_L + Rot1_L * L2_L/2);
com_link_R = pos_body + R_body * (Reflection_Y*L1_R + Rot1_R * Reflection_Y*L2_R/2);
pos_elbow_L = pos_body + R_body * (L1_L + Rot1_L * L2_L);
pos_elbow_R = pos_body + R_body *(Reflection_Y*L1_R +Rot1_R * Reflection_Y*L2_R);
com_plate_L = pos_elbow_L + R_body * Rot1_L * Rot2_L * L3_L/2;
com_plate_R = pos_elbow_R + R_body * Rot1_R * Rot2_R * Reflection_Y*L3_R/2;

% Linear velocities
vel_link_L = jacobian(com_link_L, q)*qd;
vel_link_R = jacobian(com_link_R, q)*qd ;
vel_plate_L = jacobian(com_plate_L, q)*qd; 
vel_plate_R = jacobian(com_plate_R, q)*qd ;

% local frame angular velocities
w_body= Rx(qx)'*Ry(qy)'*Rz(qz)'*[0;0;qzD] + Rx(qx)'*Ry(qy)'*[0;qyD;0] +Rx(qx)'*[qxD;0;0];

angvel_body = w_body;
angvel_link_L = Rot1_L.'*(w1_L + angvel_body);
angvel_link_R = Rot1_R.'*(w1_R + angvel_body);
angvel_plate_L = Rot2_L.'*(w2_L + angvel_link_L);
angvel_plate_R = Rot2_R.'*(w2_R + angvel_link_R);

%% Euler-Lagrangian Dynamics

% Energy formulations -----------------------------------------------------

K_lin = mb/2*(vel_body.'*vel_body) + ...
        ma/2*(vel_link_L.'*vel_link_L) + ...
        ma/2*(vel_link_R.'*vel_link_R) + ...
        mw/2*(vel_plate_L.'*vel_plate_L) + ...
        mw/2*(vel_plate_R.'*vel_plate_R);
    
% K_lin = mb/2*(vel_body.'*vel_body) + ...
%         ma/2*(vel_link_L.'*vel_link_L) + ...
%         ma/2*(vel_link_R.'*vel_link_R);

K_ang = angvel_body.'*I_body*angvel_body/2 + ...
        angvel_link_L.'*I_link*angvel_link_L/2 + ...
        angvel_link_R.'*I_link*angvel_link_R/2 + ...
        angvel_plate_L.'*I_plate*angvel_plate_L/2 + ...
        angvel_plate_R.'*I_plate*angvel_plate_R/2;

% K_ang = angvel_body.'*I_body*angvel_body/2 + ...
%         angvel_link_L.'*I_link*angvel_link_L/2 + ...
%         angvel_link_R.'*I_link*angvel_link_R/2;
%      
U = mb*g*pz + ...
    ma*g*com_link_L(3) + ...
    ma*g*com_link_R(3) + ...
    mw*g*com_plate_L(3) + ...
    mw*g*com_plate_R(3);

K=K_ang + K_lin;
 

L =K- U;

jac_K   = jacobian(K,qd);
M      = ((jacobian(jac_K',qd))');

% Coriolis matrix
tmp     = jacobian(M*qd,q)-(1/2)*jacobian(M*qd,q).'; % page 419, westervelt book
Cdq     = tmp*qd;

% Gravity matrix
G       = jacobian(U,q).';

H       = simplify(Cdq + G);

%% Aerodynamics forces

% position and velocity at the aerodynamic forces applied locations
pos_aero_L = pos_elbow_L + R_body * Rot1_L * Rot2_L * La_L;
pos_aero_R = pos_elbow_R + R_body * Rot1_R * Rot2_R * Reflection_Y*La_R;
vel_aero_L = jacobian(pos_aero_L,q)*qd;
vel_aero_R = jacobian(pos_aero_R,q)*qd;

% B matrix for aero forces (inertial force)
Ba_L = jacobian(pos_aero_L,q).';
Ba_R = jacobian(pos_aero_R,q).';


fprintf(' Completed \n')

%------------------------------------------------------------------------
%% Save functions

params_dyn = [...
            mb; ma; mw; g; L1_L ;L2_L; L3_L; L1_R; L2_R; L3_R;...
            Ib ; Ia; Iw]; 

func_dyn = Function('f',{q, qd, params_dyn},{M, Cdq,G});
function_M = Function('f',{q, qd, params_dyn},{M});
function_Cdq = Function('f',{q, qd, params_dyn},{Cdq});
function_G = Function('f',{q, qd, params_dyn},{G});
func_aero_L = Function('f',{q, qd, params_dyn,La_L},{Ba_L, vel_aero_L});
func_aero_R = Function('f',{q, qd, params_dyn,La_R},{Ba_R, vel_aero_R});

% disp(func_dyn)
% disp(func_aero_R)
% disp(func_aero_R)


opts = struct('main', true,...
              'mex', true);

func_dyn.generate('func_aerobat_dyn.c',opts);
function_M.generate('function_M.c',opts);
function_Cdq.generate('function_Cdq.c',opts);
function_G.generate('function_G.c',opts);
func_aero_L.generate('func_aerobat_aero_L.c',opts);
func_aero_R.generate('func_aerobat_aero_R.c',opts);

toc


% Note: you might need to import casadi again 
% Start with 
%   >> import casadi.*
% Compiling as C code:
% Compile with 
%   >> gcc -fPIC -shared <generated_file_name>.c -o <desired_file_name>.so
% Call using:
%   >> func = external('f', './<desired_file_name>.so');
%   >> outputs = func(inputs, params);
%
% Compiling as mex:
% Compile with 
 mex func_aerobat_dyn.c
 mex function_M.c
 mex function_Cdq.c
 mex function_G.c
 mex func_aerobat_aero_L.c
 mex func_aerobat_aero_R.c
% Call using:
%   >> func = external('f', '<generated_file_name>.mexw64');
%   >> outputs = func(inputs, params);