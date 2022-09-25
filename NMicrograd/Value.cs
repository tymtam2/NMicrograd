// C# reimplementation of https://github.com/karpathy/micrograd
// at c911406e5ace8742e5841a7e0df113ecb5d54685 Sat Apr 18 12:15:25 2020 -0700


namespace NMicrograd;

public record Value {

    public double Data {get; set;}

    public string Op;

    public List<Value> Children { get; } = new List<Value>();

    public double Grad { get; set; } = 0;

    public Action FBackward { get; set; } = () => { };

    public Value(double data) 
    {
        Data = data;
        Op = "";
    }

    private Value(double data, string op, /*Action backward,*/ List<Value> children) 
    {
        Data = data;
        Op = op;
        //Backward = backward;
        Children = children; 
    }

    public static Value operator +(Value a, Value b)
    {
        var @out = new Value(
            data: a.Data + b.Data,
            op: "+",
            children: new List<Value>() {a, b});

         //Console.WriteLine($"+: Forward");

        @out.FBackward = () => {
            var ga = a.Grad;
            var gb = b.Grad;
            a.Grad += @out.Grad;
            b.Grad += @out.Grad;

            //Console.WriteLine($"+: {ga} += {@out.Grad} => {a.Grad}");
            //Console.WriteLine($"+: {gb} += {@out.Grad} => {b.Grad}");
        };

        return @out;
    }

    public static Value operator *(Value a, Value b)
    {
        var @out = new Value(
            data: a.Data * b.Data,
            op: "*",
            children: new List<Value>() {a, b});

        //Console.WriteLine($"*: Forward");
        @out.FBackward = () => {
           var ga = a.Grad;
           a.Grad += b.Data * @out.Grad;
          
           var gb = b.Grad;
           b.Grad += a.Data * @out.Grad;

           //Console.WriteLine($"*: {ga} += {b.Data}*{@out.Grad} => {a.Grad}");
           //Console.WriteLine($"*: {gb} += {a.Data}*{@out.Grad} => {b.Grad}");
        };

        return @out;
    }

    public Value Pow( int other)
    {
        var @out = new Value(data: (double) Math.Pow(Data, other), children: new List<Value>{this}, op: $"**{other}");

        var a = this;
        @out.FBackward = () =>
        {
            a.Grad += (other * (double)Math.Pow(a.Data,(other-1))) * @out.Grad;
        };

        return @out;
    }

    public static Value operator -(Value a) => a * -1;

    public static Value operator -(Value a, Value b) => a + (-b);

    public static Value operator /(Value a, Value b) => a * (b.Pow(-1));
    
    public Value ReLU()
    {
        var @out = new Value(
            data: (Data < 0) ? 0 : Data, 
            children: new List<Value>(){this}, 
            op: "ReLU");

        var a = this;
        @out.FBackward = () => {
            var g = a.Grad;
            a.Grad += ((@out.Data > 0) ? 1 : 0) * @out.Grad;
            //Console.WriteLine($"ReLU: {g} += {@out} => {a.Grad}");
        };

        return @out;
    }

    public static implicit operator Value(double someValue) => new Value(data: someValue);

    public void Backward()  
    {
        // topological order all of the children in the graph
        var topologicallyOrdered = new List<Value>();
        var visited = new HashSet<Value>();
        
        BuildInTopologicalOrder(this);

        // go one variable at a time and apply the chain rule to get its gradient
        Grad = 1;
        topologicallyOrdered.Reverse();
        foreach(var v in topologicallyOrdered)
        {
            v.FBackward();
        }

        void BuildInTopologicalOrder(Value v)
        {
            if (!visited.Contains(v))
            {
                visited.Add(v);
                foreach(var child in v.Children)
                {
                    BuildInTopologicalOrder(child);
                }
                topologicallyOrdered.Add(v);
            }
            //Console.WriteLine($"ordered: {topologicallyOrdered.Count()}");
        }
    }   

    public override string? ToString() => $"Value(data={Data}, grad={Grad})";
}