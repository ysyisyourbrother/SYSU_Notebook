function H = HomomorphicFiltering( gamma_H, gamma_L, c, D0, height, width )
    for i = 1 : height
        x = i - (height / 2);
        for j = 1 : width
            y = j - (width / 2);
            H(i, j) = (gamma_H - gamma_L) * (1 - exp(-c * ((x ^ 2 + y ^ 2) / D0 ^ 2))) + gamma_L;
        end
    end
end