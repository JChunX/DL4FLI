function [tau1, tau2, ratio] = generate_lifetime(image, tau1_range, tau2_range)
% generate random lifetime values for the 28x28 binary image
% t1 - fast lifetime ranges: [t1_upper, t1_lower]
% t2 - slow lifetime ranges
    
    m = size(image, 1);
    n = size(image, 2);
%     Create randomly generated value matrices for the tau1 and tau2
%     thresholds of interest.
    tau1 = rand(m, n)*(tau1_range(2)-tau1_range(1)) + tau1_range(1);
    tau2 = rand(m,n)*(tau2_range(2)-tau2_range(1)) + tau2_range(1);

    tau1 = tau1.*image;
    tau2 = tau2.*image;
    ratio = rand(m, n).*image;
end