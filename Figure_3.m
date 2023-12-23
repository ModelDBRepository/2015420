clear all
close all
clc

%% Simulation Parameters/Context Changes 
t1 = 500; %US start 
t2 = 900; %FS start 
FSINT = 30; %FS duration 
t3 = t2 + FSINT*13; %FS and US end 
t4 = t3 + t1; %start a second US 
T = t4 + 400; %total time 

%% 
dt = 0.001; %integration time step 
nt = round(T/dt); 
N = 100; %number of neurons 
tm = 0.01; %membrane time constant 
td = 0.02; %decay time in seconds 
tr = 0.002; %rise time in seconds 
tref = 0.002; %refractory time period 
tcar = 0.1; %calcium rise time 
tcad = 1.0; %calcium decay time. 
vreset = -65; %voltage reset 
vpeak = -40; %voltage peak. 
BIAS = vpeak-1; %bias current 
v = vreset + (vpeak-vreset)*rand(N,1);
tspike = zeros(nt,2);
IPSC = zeros(N,1); %post synaptic current storage variable 
h = zeros(N,1); %Storage variable for filtered firing rates
r = zeros(N,1); %second storage variable for filtered rates 
hr = zeros(N,1); %Third variable for filtered rates 
hslow = zeros(N,1); %Storage variable for filtered firing rates
rslow = zeros(N,1); %second storage variable for filtered rates 
hrslow = zeros(N,1); %Third variable for filtered rates 
storev = zeros(nt,10);
storer = zeros(nt,10);
JD = 0*IPSC; %storage variable required for each spike time 
tlast = zeros(N,1); %storage variable for refractory spike times 
ns = 0;% total number of spikes fires 
REC = zeros(T/0.1,N); %store calcium signal. 
sigma = 0.5; %noise variance.  
w = zeros(N,1); %initial adaptation variable 
d = 20; %adaptation increment 
tw = 0.1;   %adaptation time constant. 
kd = 0;  %storage initialization 
%% input parameters 
Nin = 1000; %number of context inputs. 
omegaFS= 2+randn(N,1); %FS input weight.  
omega = randn(N,Nin)/sqrt(Nin);  %weights for context inputs.
omega0 = omega;
mO = max(mean(omega0'))*1.2 %set the max weight sum. 
qin = zeros(Nin,1);
nu = 1+zeros(Nin,1);

%% FS stimulus. 
FS = zeros(nt,1); 
time = (1:nt)*dt;
for j = 1:10
FS = FS + exp(-((1:nt)*dt-(t2+j*FSINT)).^2)';
end



%% plasticity parameters 
rthresh = 2.5;
epsilon = 1.4e-2; 
plast = 1; 
tau_aux = 400; 
aux = zeros(N,Nin);
iax = randperm(N*Nin,10); 
store_aux = zeros(nt,10);

%% run integration
for i = 1:nt 


%% create Poisson/stochastic input spikes. 
if (dt*i>t1)&(dt*i<t3)|(dt*i>t4)
    nu(1:Nin/2) = 8; else nu(1:Nin/2) = 1;
end
q = rand(Nin,1); 
qin = qin  + (q<nu*dt);
qin = qin*exp(-dt/td); %filter spikes. 
    
%% Integrate neurons and store spikes.    
IPSC = omega*qin;
IFS = omegaFS*FS(i);
I = BIAS + IPSC + IFS - w; %Current to Neurons 
dv = (dt*i>tlast + tref).*(( (-v+I)/tm)); %Voltage equation with refractory period 
v = v + dt*(dv) + sqrt(dt)*sigma*(dt*i>tlast + tref).*randn(N,1)/sqrt(tm); 
w = w + dt*(-w/tw) + d*(v>=vpeak); %apply adaptation variable. 

index = find(v>=vpeak);  %Find the neurons that have spiked 
if length(index)>0
tspike(ns+1:ns+length(index),:) = [index,0*index+dt*i]; %store spike times 
ns = ns + length(index);
end

% Filtered version of neuronal voltages 
if tcar == 0  
rslow = rslow*(1-dt/tcar)+ (v>=vpeak)/tcar;
else
rslow = rslow*(1-dt/tcar) + hrslow*dt; 
hrslow = hrslow*(1-dt/tcad) + (v>=vpeak)/(tcar*tcad);
end



%% reset spikes and apply refractory period. 
tlast = tlast + (dt*i -tlast).*(v>=vpeak);  %Used to set the refractory period of LIF neurons 
v = v + (30 - v).*(v>=vpeak); %rest the voltage and apply a cosmetic spike.  
storev(i,:) = v(1:10);
storer(i,:) = rslow(1:10);
v(v>=vpeak) = vpeak;

store_aux(i,:) = aux(iax);

if mod(i,0.1/dt)==1 
kd = kd + 1; 
REC(kd,:) = rslow;
end




%% apply the plasticity rule. 
if plast == 1
i0 = find(rslow>rthresh);
if length(i0)>0 
aux(i0,:)  = aux(i0,:) + dt*((mO-mean(omega(i0,:)')').*(rslow(i0))*ones(1,Nin))/tau_aux ;
end
end
aux = aux*exp(-dt/tau_aux);
omega = omega + aux*dt*epsilon;



v = v + (vreset - v).*(v>=vpeak); %reset spike time 

if mod(i,1000)==1
i/nt
end

end
%% 


 close all
 str = datestr(now,30);
 save(sprintf('%s.mat',str))
%% Perform analysis on intervals. 
tlong = (1:1:kd)*T/kd; 
hc1 = find(tlong<t1-5&tlong>5);
hc2 = find(tlong>t3+5&tlong<t4-5); 
us1 = find(tlong<t2-5&tlong>t1+5); 
us2 = find(tlong>t4+5&tlong<T-5); 
fs1 = find(tlong<t3-5&tlong>t2+5); 


%% Plot Results 
figure(1)
US_SCORE1 = trapz(REC(round(t1/0.1):round(t2/0.1),:))/(t2-t1);
US_SCORE2 = trapz(REC(round(t4/0.1):round(T/0.1),:))/(T-t4);
[rho,pval] = corr(US_SCORE1',US_SCORE2')
plot(US_SCORE1,US_SCORE2,'k.','MarkerSize',14), hold on 
plot([0,20],[0,20],'b')
xlim([0,max(US_SCORE1)])
ylim([0,max(US_SCORE2)])
xlabel('US Score')
ylabel('FRD1 Score') 
title('Area Under Curve Score')

figure(2)
subplot(1,4,1) 
imagesc(tlong(us1),1:N,REC(us1,:)',[0,2.4])
xlim([550,750])
title('US')
subplot(1,4,2)
imagesc(tlong(fs1),1:N,REC(fs1,:)',[0,2.4])
xlim([tlong(fs1(1)),tlong(fs1(1))+30*11])
title('FS')
subplot(1,4,3)
imagesc(tlong(us2),1:N,REC(us2,:)',[0,2.4])
colormap('jet')
xlim([1900,2100])
title('FRD1')
subplot(1,4,4)
plot(tlong-t1,mean(REC')), hold on 
plot(tlong-t4,mean(REC'))
xlim([100,300])
legend('US','FRD1')
title('US Changes')
%% Apply SVD/PCA. 
[u,s,v] = svds(REC,3); 
%% 
u(:,1) = -u(:,1); 
u(:,2) = -u(:,2);
%%

figure(3) 
tlong = (1:1:kd)*T/kd; 
hc1 = find(tlong<t1-5&tlong>5);
hc2 = find(tlong<t4-5&tlong>t3+5);
us1 = find(tlong<t2-5&tlong>t1+5); 
us2 = find(tlong>t4+5&tlong<T-5); 
fs1 = find(tlong<t3-5&tlong>t2+5); 
dlong = tlong(2)-tlong(1); 
plot(u(hc1,1),u(hc1,2),'k','LineWidth',2), hold on 
plot(mean(u(hc1,1)),mean(u(hc1,2)),'g.','MarkerSize',14), hold on 
plot(u(us1,1),u(us1,2),'r','LineWidth',2) 
plot(mean(u(us1,1)),mean(u(us1,2)),'g.','MarkerSize',14) 
xlabel('PC1')
ylabel('PC2')
    for k = 1:10 
  i0 = (t2+k*FSINT-2):dlong:(t2+k*FSINT+2);
  i0 = round(i0/dlong); 
  
plot(u(i0,1),u(i0,2),'m','LineWidth',2) 

plot(u(i0(1),1),u(i0(2),2),'k.','MarkerSize',14) 
plot(u(i0(end),1),u(i0(end),2),'r.','MarkerSize',14) 
if k ==1 
plot(u(us2,1),u(us2,2),'b','LineWidth',2)
end
    end
plot(mean(u(us2,1)),mean(u(us2,2)),'g.','MarkerSize',14)

q1 = [mean(u(us1,1)),mean(u(us2,1))];
q2 = [mean(u(us1,2)),mean(u(us2,2))];

slope = (q2(2)-q2(1))/(q1(2)-q1(1));
y = q2(1) + (q2(2)-q2(1))*(0:0.01:7); 
x = q1(1) + (q1(2)-q1(1))*(0:0.01:7); 

 plot(x,y,'g','LineWidth',2)  
