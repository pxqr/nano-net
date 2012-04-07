using System;

namespace Nanon.Math.Linear
{	
	public class Normalization
	{
		Vector avgs;  // averages
		Vector stds;  // standart derivation
		Vector istds; // inverted standart derivation
		
		public Normalization(int size)
		{
			avgs = new Vector(size);
			avgs.Transform(x => 1.0d);
			stds = new Vector(size);
			stds.Transform(x => 1.0d);
		}
		
		public Normalization(Vector[] vectors)
		{
			if (vectors.Length == 0)
				throw new ArgumentException("Can't normalize empty set.");
			
			var setSize = vectors.Length;
			var featureCount = vectors[0].Cells.Length;
			
			avgs = new Vector(featureCount);
			stds = new Vector(featureCount);
			
			// find avgs and stds over vectors
			for (var feature = 0; feature < featureCount; ++feature)
			{
				var avgAcc = 0.0d;
				var savgAcc = 0.0d;
				
				for (var i = 0; i < setSize; ++i)
				{
					var x = vectors[i].Cells[feature];
					avgAcc += x;
					savgAcc += x * x;
				}
				
				var avg = avgAcc / (double)setSize;
				var std = System.Math.Sqrt((savgAcc / (double)setSize) - avg * avg);
				
				// std may cause overflow 
				if (std <= 0)
					std = 1.0d;
				
				avgs[feature] = avg; 
				stds[feature] = std;
			}
			
			istds   = stds.Map(x => 1.0d / x);
		}
		
		public void Normalize(Vector vector)
		{			
			vector.Sub(avgs, vector);       // shift
			vector.Multiply(istds, vector); // scale
		}
		
		public void Denormalize(Vector vectors)
		{
			throw new NotImplementedException();	
			//return (vector + avgs) * stds;
		}
	}
}

