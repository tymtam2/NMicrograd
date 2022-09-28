// C# reimplementation of https://github.com/karpathy/micrograd
// at c911406e5ace8742e5841a7e0df113ecb5d54685 Sat Apr 18 12:15:25 2020 -0700

namespace NMicrograd;

public abstract class Module
{
    public void ZeroTheGrads()
    {
        foreach (var p in GetParameters())
            p.Grad = 0;
    }

    public abstract IEnumerable<Value> GetParameters();
}

public class Neuron : Module
{
    private static Random random = new(Seed: 42);

    public List<Value> W { get; }

    public Value B { get; }

    public bool Nonlin { get; }

    public Neuron(int nin, bool nonlin = true)
    {
        W = Enumerable.Range(0, nin).Select(i => new Value(-1 + 2 * random.NextSingle())).ToList();
        B = new Value(0);
        Nonlin = nonlin;
    }

    public Value F(IEnumerable<Value> x)
    {
        // Or: 
        //  var act = B;
        //  foreach(var wixi in W.Zip(x, (wi,xi) => wi*xi)) 
        //  act += wixi;
        var act = W.Zip(x, (wi, xi) => wi * xi).Aggregate(seed: B, (s, d) => s + d);

        return Nonlin ? act.ReLU() : act;
    }

    public override IEnumerable<Value> GetParameters() => W.Append(B);

    public override string? ToString() => $"{(Nonlin ? "ReLU" : "Linear")}Neuron({W.Count})";
}

public class Layer : Module
{
    public List<Neuron> Neurons { get; }

    public Layer(int nin, int nout, bool nonlin = true)
    {
        Neurons = Enumerable.Range(0, nout).Select(i => new Neuron(nin, nonlin)).ToList();
    }

    public IEnumerable<Value> F(IEnumerable<Value> x)
    {
        var @out = Neurons.Select(n => n.F(x));
        //Console.WriteLine($"< Layer F, act = {string.Join(",", @out.Select( x => $"{x.Data:0.000}"))}");   
        return @out; // In Python they can change return types:  @out[0] if len(out) == 1 else out
    }

    public override IEnumerable<Value> GetParameters() => Neurons.SelectMany(n => n.GetParameters());

    public override string? ToString() => $"Layer of [{string.Join(",", Neurons)}]";
}

public class MLP : Module
{
    public List<Layer> Layers { get; }

    public MLP(int nin, IEnumerable<int> nouts)
    {
        var sz = new List<int> { nin };
        sz.AddRange(nouts);
        Layers = Enumerable.Range(0, nouts.Count())
            .Select(i => new Layer(nin: sz[i], nout: sz[i + 1], nonlin: i != nouts.Count() - 1))
            .ToList();
    }

    public IEnumerable<Value> F(IEnumerable<Value> x)
    {
        foreach (var layer in Layers)
            x = layer.F(x);
        return x!;
    }

    public override IEnumerable<Value> GetParameters() => Layers.SelectMany(n => n.GetParameters());

    public override string? ToString() => $"MLP of [{string.Join(",", Layers.Select(l => l.ToString()))}]";
}