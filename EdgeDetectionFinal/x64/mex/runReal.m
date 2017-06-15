function res = runReal(I,prm)
    close all;
    
    tic;
    [m,n] = size(I);
    R = zeros(size(I));
    block = 2^floor(log2(min(size(I))))+1;
    block = min(block,129);
    disp(block);
    gap = 10;
    
    xGrid = getGrid(block,gap,m);
    yGrid = getGrid(block,gap,n);
    
    iter = length(xGrid)*length(yGrid);
    disp(iter);
    fprintf('Number of Iterations = %d\n',iter);
    counter = 1;
    for x0 = xGrid
        for y0 = yGrid
            disp(counter);
            counter = counter+1;
            curI = I(x0:x0+block-1,y0:y0+block-1);
            curR = run(curI,prm);
            
            curR(1:2,:) = 0;
            curR(end-1:end,:) = 0;
            curR(:,1:2) = 0;
            curR(:,end-1:end) = 0;
            
            R(x0:x0+block-1,y0:y0+block-1) = max(R(x0:x0+block-1,y0:y0+block-1),curR);
        end
    end
    
    res = R;
end

function grid = getGrid(block,gap,max)
    grid = zeros(1,max);
    grid(1) = 1;
    for i= 2:max
         grid(i)= grid(i-1)+block-gap;
         if grid(i)+block-1>max
            grid(i) = max-block+1;
            break;
         end
    end
    grid(grid == 0) = [];
    
    if length(grid)>=2 && grid(end) == grid(end-1)
        grid(end) = [];
    end
    
end
