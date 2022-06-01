%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Aerobat equation of motion symbolic derivations
%
% 8-7-2020
% Author: Eric Nauli Sihite
% Northeastern University, Boston, USA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Setup

close all, clear all, clc

% addpath('generated_functions\') % filesep makes this usable on all OS
addpath(['generated_functions',filesep]) 
addpath(['Utility_functions',filesep])    

%% Variable Declarations

% constrant parameter declarations ----------------------------------------

% Mass scalar variables (constant)
syms mb ma mw g t real

% Length vector variables (constant)
L1_L = sym('L1_L_', [3,1], 'real');   % body to shoulder (left)
L2_L = sym('L2_L_', [3,1], 'real');   % shoulder to elbow (left)
L3_L = sym('L3_L_', [3,1], 'real');   % elbow to wingtip (left)
L1_R = sym('L1_R_', [3,1], 'real');   % body to shoulder (right)
L2_R = sym('L2_R_', [3,1], 'real');   % shoulder to elbow (right)
L3_R = sym('L3_R_', [3,1], 'real');   % elbow to wingtip (right) 

% Aerodynamics length vector variables (elbow to aero force location)
La_L = sym('La_L_', [3,1], 'real');
La_R = sym('La_R_', [3,1], 'real');

% Inertia variables (constant)
Ib = sym('Ib_', [3,1], 'real'); 
Ia = sym('Ia_', [3,1], 'real'); 
Iw = sym('Iw_', [3,1], 'real'); 
I_body = diag (Ib);    % body inertia matrix
I_link = diag (Ia);    % link inertia matrix
I_plate = diag (Iw);   % plate inertia matrix

% time varying variables declarations -------------------------------------

% Body degrees of freedom
px=sym('px');
py=sym('py');
pz=sym('pz');

qx=sym('qx');
qy=sym('qy');
qz=sym('qz');

qp_L=sym('qp_L');
qm_L=sym('qm_L');
qe_L=sym('qe_L');
qf_L=sym('qf_L');

qp_R=sym('qp_R');
qm_R=sym('qm_R');
qe_R=sym('qe_R');
qf_R=sym('qf_R');

pos_body = [px;py;pz];    % body CoM linear position
angles = [qx;qy;qz];    % body angles



% Note: only longitudinal dynamic is considered 
% R_body=rot_z(qz)*rot_y(qy)*rot_x(qx); 
R_body = rot_y(qy); 



pxD=sym('pxD');
pyD=sym('pyD');
pzD=sym('pzD');

qxD=sym('qxD');
qyD=sym('qyD');
qzD=sym('qzD');

qpD_L=sym('qpD_L');
qmD_L=sym('qmD_L');
qeD_L=sym('qeD_L');
qfD_L=sym('qfD_L');

qpD_R=sym('qpD_R');
qmD_R=sym('qmD_R');
qeD_R=sym('qeD_R');
qfD_R=sym('qfD_R');

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
Rot1_L = rot_z(-qm_L)*rot_x(qp_L); % plunge then medio
Rot2_L = rot_x(qe_L)*rot_y(-qf_L); % feather then elbow extension
Rot1_R = rot_z(qm_R)*rot_x(-qp_R);
Rot2_R = rot_x(-qe_R)*rot_y(-qf_R);

% Angular velocities (about their base frame)
w1_L = [0;0;-qmD_L] + rot_z(-qm_L)*[qpD_L;0;0];
w2_L = [qeD_L;0;0] + rot_x(qe_L)*[0;-qfD_L;0];

w1_R = [0;0;qmD_R] + rot_z(qm_R)*[-qpD_R;0;0];
w2_R = [-qeD_R;0;0] + rot_x(-qe_R)*[0;-qfD_R;0];



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
w_body= rot_x(qx)'*rot_y(qy)'*rot_z(qz)'*[0;0;qzD] + rot_x(qx)'*rot_y(qy)'*[0;qyD;0] +rot_x(qx)'*[qxD;0;0];

angvel_body = w_body;
angvel_link_L = Rot1_L.'*(w1_L + angvel_body);
angvel_link_R = Rot1_R.'*(w1_R + angvel_body);
angvel_plate_L = Rot2_L.'*(w2_L + angvel_link_L);
angvel_plate_R = Rot2_R.'*(w2_R + angvel_link_R);

%% Euler-Lagrangian Dynamics

% Energy formulations -----------------------------------------------------

% NOTE: we assume a 1-link wing when computing kinetic energy


K_lin = mb/2*(vel_body.'*vel_body) + ...
        ma/2*(vel_link_L.'*vel_link_L) + ...
        ma/2*(vel_link_R.'*vel_link_R) + ...
        mw/2*(vel_plate_L.'*vel_plate_L) + ...
        mw/2*(vel_plate_R.'*vel_plate_R);
% 
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
     
U = mb*g*pz + ...
    ma*g*com_link_L(3) + ...
    ma*g*com_link_R(3) + ...
    mw*g*com_plate_L(3) + ...
    mw*g*com_plate_R(3);

K=K_ang + K_lin;
 

L =K- U;
% Inertia matrix
jac_K   = jacobian(K,qd);
M      = ((jacobian(jac_K',qd))');

% Coriolis matrix
% tmp     = jacobian(M*qd,q)-(1/2)*jacobian(M*qd,q).'; % page 419, westervelt book
% Cdq     = tmp*qd;

% Gravity matrix
% G       = jacobian(U,q).';
% 
% H       = simplify(Cdq + G);

%% Aerodynamics forces

% position and velocity at the aerodynamic forces applied locations
pos_aero_L = pos_elbow_L + R_body * Rot1_L * Rot2_L * La_L;
pos_aero_R = pos_elbow_R + R_body * Rot1_R * Rot2_R * Reflection_Y*La_R;
vel_aero_L = jacobian(pos_aero_L,q)*qd;
vel_aero_R = jacobian(pos_aero_R,q)*qd;

% B matrix for aero forces (inertial force)
Ba_L = jacobian(pos_aero_L,q).';
Ba_R = jacobian(pos_aero_R,q).';


%% Generate Matlab Function

% System states
states = [q ;qd];

% System parameters
params = [ mb; ma; mw; g; Ib; Ia; Iw; ...
          L1_L; L2_L; L3_L; L1_R; L2_R; L3_R];
      
% NOTE: using filesep makes folder names usable on all OS
% 
% Dynamic equation of motion for the massed system
matlabFunction(M, 'File', ...
  ['generated_functions', filesep, 'func_M'], 'Vars', {states, params},...
    'Optimize',false);

% matlabFunction(Cdq, 'File', ...
%   ['generated_functions', filesep, 'func_Cdq'], 'Vars', {states, params},...
%     'Optimize',false);

% matlabFunction(G, 'File', ...
%   ['generated_functions', filesep, 'func_G'], 'Vars', {states, params},...
%     'Optimize',false);
  
% matlabFunction(M, h, 'File', ...
%   ['generated_functions', filesep, 'func_Mh'], 'Vars', {states, params},...
%     'Optimize',true);

matlabFunction(Ba_L, vel_aero_L, 'File', ...
  ['generated_functions', filesep, 'func_Ba_L'], ...
    'Vars', {states, params, La_L},...
    'Optimize',false);
  
matlabFunction(Ba_R, vel_aero_R, 'File', ...
  ['generated_functions', filesep, 'func_Ba_R'], ...
    'Vars', {states, params, La_R},...
    'Optimize',false);
  
  
disp('done!')












