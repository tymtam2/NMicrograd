// C# reimplementation of https://github.com/karpathy/micrograd
// at c911406e5ace8742e5841a7e0df113ecb5d54685 Sat Apr 18 12:15:25 2020 -0700


using NMicrograd;

var model = new MLP(2, new []{16, 16, 1}); // 2-layer neural network
//Console.WriteLine(model);
//Console.WriteLine($"Number of parameters: {model.GetParameters().Count()}");

// Please see the notebook in the `data` folder for the steps how these were captures
var lines = File.ReadAllLines("NMicrograd.Demo.Console\\data\\labeled_data.csv");
// Format is x1, x2, label
var data = lines.Select( l => l.Split(',')) 
   .Select( a => new {
        X = new []{double.Parse(a[0]), double.Parse(a[1])},
        Y = int.Parse(a[2])});

// Copies exact weights used in the https://github.com/karpathy/micrograd/blob/master/demo.ipynb (from 14 Apr 2020)
// so that the descent is exactly the same
// Please see the notebook in the `data` folder for the steps how these were captures
var copyInitialWeightsFromTheOriginalDemo = true;
if(copyInitialWeightsFromTheOriginalDemo)
{
    var weights = File.ReadAllText("NMicrograd.Demo.Console\\data\\weights_data.csv")
        .Split(',')
        .Select( s => double.Parse(s));

    if(model.GetParameters().Count() != weights.Count())
    {
        throw new Exception($"Bad weights. Has {weights.Count()}, need {model.GetParameters().Count()}");
    }

    // Fill the weights
    foreach((var p, var w) in model.GetParameters().Zip(weights, (p,w) => (p,w)))
        p.Data = w; 
}


// optimization
for(int k = 0; k<100; k++)
{
    // forward
    (var total_loss, var acc) = GetLoss(
        x: data.Select(d => d.X), 
        y: data.Select(d => d.Y),
        model);
    
    // backward
    model.ZeroTheGrads();
    total_loss.Backward();
    
    // update (stochastic gradient descent)
    var learning_rate = 1.0 - (0.9*k)/100;
    foreach(var p in model.GetParameters())
        p.Data -= learning_rate * p.Grad;

    if (k % 1 == 0)
        Console.WriteLine($"step {k} loss {string.Format("{0:F17}", total_loss.Data)}, accuracy {acc*100:0.0}%");
}


static (Value TotalLoss, double Accuracy) GetLoss(
    IEnumerable<IEnumerable<double>> x,
    IEnumerable<int> y,
    MLP model)
{
    var inputs = x.Select( d => d.Select( x => new Value(data: x)));

    // forward the model to get scores
    var scores = inputs.Select(input => model.F(input).First());

    // svm "max-margin" loss
    var losses = y.Zip(scores, (yi, scorei) => (1 + -yi*scorei).ReLU());
    
    // var data_loss = new Value(0);
    // foreach(var l in losses) data_loss+=l;
    // data_loss = data_loss * (1.0 / losses.Count());
    var data_loss = losses.Aggregate(new Value(0), (s,d) => s+d) * (1.0 / losses.Count());

    // L2 regularization
    var alpha = 1e-4;
    //var reg_loss = new Value(0);
    //model.GetParameters().Select(p => p*p).ToList().ForEach(p2 => reg_loss += p2); 
    //reg_loss = alpha * reg_loss;
    var reg_loss = alpha * model.GetParameters().Aggregate(new Value(0), (s,d) => s+(d*d));
    var total_loss = data_loss + reg_loss;
    
    // also get accuracy
    var accuracy = y.Zip(scores, (yi, scorei) => (yi > 0) == (scorei.Data > 0));
    return (TotalLoss: total_loss, Accuracy: accuracy!.Sum(a => a ? 1.0 : 0) / accuracy!.Count());
}