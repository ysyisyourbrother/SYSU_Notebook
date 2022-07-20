function mat = Centralize( mat )
    [height, width] = size(mat);
    for i = 1 : height
        for j = 1 : width
            if mod(i + j, 2) == 1
                mat(i, j) = -mat(i, j); % -1^(i+j)
            end
        end
    end
end