// C# reimplementation of https://github.com/karpathy/micrograd
// at c911406e5ace8742e5841a7e0df113ecb5d54685 Sat Apr 18 12:15:25 2020 -0700

namespace NMicrograd.Tests;

[TestClass]
public class UnitTests
{
    [TestMethod]
    public void TestSanityCheck()
    {
        var x = new Value(-4.0f);
        var z = 2 * x + 2 + x;
        var q = z.ReLU() + z * x;
        var h = (z * z).ReLU();
        var y = h + q + q * x;
        y.Backward();
        var xmg = x;
        var ymg = y;

        // The original micrograd values are
        // Value(data=-4.0, grad=46.0)
        // Value(data=-20.0, grad=1)

        AssertEqual(new Value(data: -4.0) { Grad = 46.0 }, xmg);
        AssertEqual(new Value(data: -20.0) { Grad = 1 }, ymg);
    }

    [TestMethod]
    public void test_more_ops()
    {
        var a = new Value(-4.0f);
        var b = new Value(2.0f);
        var c = a + b;
        var d = a * b + b.Pow(3);
        c += c + 1;
        c += 1 + c + (-a);
        d += d * 2 + (b + a).ReLU();
        d += 3 * d + (b - a).ReLU();
        var e = c - d;
        var f = e.Pow(2);
        var g = f / 2.0f;
        g += 10.0f / f;
        g.Backward();
        var (amg, bmg, gmg) = (a, b, g);

        // The original micrograd values are
        // Value(data=-4.0, grad=138.83381924198252), 
        // Value(data=2.0, grad=645.5772594752186), 
        // Value(data=24.70408163265306, grad=1)

        AssertEqual(new Value(data: -4.0) { Grad = 138.83381924198252 }, amg);
        AssertEqual(new Value(data: 2.0) { Grad = 645.5772594752186 }, bmg);
        AssertEqual(new Value(data: 24.70408163265306) { Grad = 1 }, gmg);
    }

    public static void AssertEqual(Value expected, Value actual)
    {
        Assert.AreEqual(expected.Data, actual.Data, float.Epsilon, $"{expected} != {actual}");
        Assert.AreEqual(expected.Grad, actual.Grad, $"{expected} != {actual}");
    }
}