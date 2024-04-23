function [X1]=EGolden(X,N,dim,best_position,hiddennum,net,p_train,t_train)
    a=min(best_position);
    b=max(best_position);
    for i=1:N
        for j=1:dim
            op(i,j)=rand*(a+b)-best_position(1,j);
            if op(i,j)>b || op(i,j)<a
                op(i,j)=a+(b-a)*rand;
            end
        end
        fit=fun(X(i,:),hiddennum,net,p_train,t_train);
        newfit=fun(op(i,:),hiddennum,net,p_train,t_train);
        if newfit<fit
            X(i,:)=op(i,:);
        end
    end
    X1=X;
end


