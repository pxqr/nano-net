using System;

namespace Nanon.Math.Linear
{
	public interface IMatrix<T>
	{
		Vector Unwind { get; }
		T ZeroCopy();
		T Copy();
		
		double Sum { get; }
		
		void Convolve(T kernel, T res);
		void Deconvolve(T kernel, T res);
		void InvolveAdd(T outputs, T kernel);
		
		void DownsampleBy2(T res);
		void UpsampleBy2(T res);
		
		void Downsample(T res);
		void Upsample(T res);
		
		
		void Add(double val, T res);
		void Sub(double val, T res);
		void Add(T lhs, T res);
		void Sub(T lhs, T res);
		void Mul(double val, T res);
		void Mul(T lhs, T res);
			
		void Transform(Func<double, double> f, T res);
		void Transform(Func<double, double> f);
		
		void SetToZero();
	}
}

