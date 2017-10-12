function plotTrainingAccuracy(info)

persistent plotObj

if mod(info.Iteration,10) == 0
    if info.State == "start"
        plotObj = animatedline;
        xlabel("Iteration")
        ylabel("Training Accuracy")
    elseif info.State == "iteration"
        addpoints(plotObj,info.Iteration,info.TrainingAccuracy)
        drawnow limitrate nocallbacks
    end

end