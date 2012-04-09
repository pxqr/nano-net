using Nanon.Model;
using Nanon.Model.Regression;
using Nanon.Math.Linear;

namespace Nanon.Statistics.Linear
{
	public class LinearRegression //: IHypothesis<Vector, double>, IRegression<Vector, double>
	{
		Vector weights;
		
		public LinearRegression(int inputSize)
		{
			weights = new Vector(inputSize + 1);
			weights.Cells[0] = 1.0d;
		}
		
		#region IRegression[Vector,System.Double] implementation
		
		public double Predict(Vector input)
		{ 
			return weights.DotProductWithBias(input);
		}
		
		#endregion

		#region IHypothesis[Vector, System.Double] implementation
		
		public double Cost(Vector input, double output)
		{
			var prediction = Predict(input);
			var sqrtCost   = prediction - output;
			return 0.5 * sqrtCost * sqrtCost;
		}
		
		public Vector[] Gradient(Vector input, double output)
		{
			var y = Predict(input);
			var grad = (y - output) * Vector.Prepend(input);
			return new Vector[] { grad };
		}

		public void Correct(Vector[] gradient)
		{
			weights += gradient[0];
		}

		public Vector[] ZeroGradient {
			get 
			{
				return new Vector[] { weights.ZeroCopy() };
			}
		}
		
		#endregion
	}
}