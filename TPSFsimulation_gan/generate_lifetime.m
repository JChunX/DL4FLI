function [tau1, tau2, ratio] = generate_lifetime(tau1_range, tau2_range)

    tau1 = (tau1_range(1) + (tau1_range(2)-tau1_range(1)).*rand);
    tau2 = (tau2_range(1) + (tau2_range(2)-tau2_range(1)).*rand);
    ratio = rand;
end