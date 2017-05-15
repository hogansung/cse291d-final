clear;
% exp0, same signals, different sampling rates
t1 = 0:0.2:12;
t2 = 0:0.3:12;
y1 = sin(2*pi*0.1*t1);
y2 = sin(2*pi*0.1*t2);
figure
plot(t1,y1,'x',t2,y2,'o')
[dtw0,C0] = DTW(y1,y2);
[P0] = OWP(dtw0);
disp(dtw0(end))
figure
plotmatrix(P0)
% exp1, shift, different sampling rates

t1 = 0+1:0.2:12+1;
t2 = 0:0.3:12;
y1 = sin(2*pi*0.1*t1);
y2 = sin(2*pi*0.1*t2);
figure
plot(t1,y1,'x',t2,y2,'o')
[dtw1,C1] = DTW(y1,y2);
[P1] = OWP(dtw1);
disp(dtw1(end))
figure
plotmatrix(P1)

% exp2, magnitude, different sampling rates

t1 = 0:0.2:12;
t2 = 0:0.3:12;
y1 = 5*sin(2*pi*0.1*t1);
y2 = sin(2*pi*0.1*t2);
figure
plot(t1,y1,'x',t2,y2,'o')
[dtw2,C2] = DTW(y1,y2);
[P2] = OWP(dtw2);
disp(dtw2(end))
figure
plotmatrix(P2)

% exp3 

t1 = 0:0.2:12;
t2 = 0:0.3:12;
y1 = (1-exp(-0.5*t1));
y2 = 2*exp(-(t2-4).^2);
figure
plot(t1,y1,'x',t2,y2,'o')
[dtw3,C3] = DTW(y1,y2);
[P3] = OWP(dtw3);
disp(dtw3(end))
figure
plotmatrix(P3)


