#include "vecmath.h"
#include <stdio.h>

template <size_t WIDTH, size_t HEIGHT>
using Mtx = BaseMtx<float, WIDTH, HEIGHT>;

template <size_t SIZE>
using MtxSq = BaseMtxSq<float, SIZE>;

template <size_t SIZE>
using Vec = BaseVec<float, SIZE>;

// The degree of the curve to fit the data points to
// 0 = constant, 1 = linear, 2 = quadratic, 3 = cubic, etc.
#define DEGREE() 1

struct DataPoint
{
	float x = 0.0f;
	float y = 0.0f;
	float weight = 1.0f;
};

int main(int argc, char** argv)
{
	DataPoint dataPoints[] =
	{
		{0.0f, 0.0f, 1.0f},
		{1.0f, 10.0f, 1.0f},
		{2.0f, 2.0f, 1.0f},
	};

	std::array<float, 2> test;
	auto a = test.size();

	// Create the W weight matrix, that has weight_i at W_ii
	static const int c_numPoints = _countof(dataPoints);
	MtxSq<c_numPoints> W{};
	for (int i = 0; i < c_numPoints; ++i)
		W[i][i] = dataPoints[i].weight;

	// Create the A matrix, where each row is a power of x, x^n where N is from 0 up to and including DEGREE()
	// Also the A^T transpose.
	Mtx<DEGREE() + 1, c_numPoints> A;
	for (int y = 0; y < c_numPoints; ++y)
		for (int x = 0; x < DEGREE() + 1; ++x)
			A[y][x] = powf(dataPoints[y].x, float(x));
	auto AT = Transpose(A);

	// Make the Y vector, which is just the y values of each point
	Vec<c_numPoints> Y;
	for (int i = 0; i < c_numPoints; ++i)
		Y[i] = dataPoints[i].y;

	// Calculate A^T * W * A
	// Also calculate A^T * W * Y
	auto ATW = Multiply(AT, W);
	auto ATWA = Multiply(ATW, A);
	auto ATWY = Multiply(ATW, Y);

	// Make an augmented matrix where the matrix A^T*W*A is on the left side, and the vector A^T*W*Y is on the right side.
	Mtx<Columns(ATWA) + 1, DEGREE() + 1> augmentedMatrix;
	for (size_t iy = 0; iy < Rows(ATWA); ++iy)
	{
		for (size_t ix = 0; ix < Columns(ATWA); ++ix)
			augmentedMatrix[iy][ix] = ATWA[iy][ix];

		augmentedMatrix[iy][Columns(ATWA)] = ATWY[iy];
	}

	// Solve the equation for x:
	// A^TWA * x = A^TWY
	// x is the coefficients of the polynomial of our fit
	GaussJordanElimination(augmentedMatrix);

	// Get our coefficients out
	Vec<DEGREE() + 1> coefficients;
	for (size_t i = 0; i < DEGREE() + 1; ++i)
		coefficients[i] = augmentedMatrix[i][Columns(ATWA)];

	// Show the equation. Note that the first coefficient is the constant, then degree 1, then degree 2, etc, so iterate in reverse order
	bool first = true;
	printf("y = ");
	for (int degree = DEGREE(); degree >= 0; --degree)
	{
		if (!first)
			printf(" + ");

		if (degree > 0 && coefficients[degree] == 0.0f)
		{
			first = true;
			continue;
		}

		if (degree > 1)
			printf("%0.2fx^%i", coefficients[degree], degree);
		else if (degree == 1)
			printf("%0.2fx", coefficients[degree]);
		else
			printf("%0.2f", coefficients[degree]);

		first = false;
	}

	// Show how close the polynomial is to the data points given.
	printf("\n\n");
	for (int i = 0; i < c_numPoints; ++i)
	{
		float y = 0.0f;
		float x = 1.0f;
		for (int degree = 0; degree <= DEGREE(); ++degree)
		{
			y += coefficients[degree] * x;
			x *= dataPoints[i].x;
		}

		printf("data[%i]: (%f, %f) weight %0.2f, equation gives %f.  Error = %f\n", i, dataPoints[i].x, dataPoints[i].y, dataPoints[i].weight, y, y - dataPoints[i].y);
	}

	// Convert from power series polynomial to bernstein form aka a Bezier curves
	printf("\nBezier Control Points = [ ");
	Vec<DEGREE() + 1> controlPoints;
	{
		// Divide by binomial coefficients
		for (int i = 0; i < DEGREE() + 1; ++i)
			coefficients[i] /= BinomialCoefficient<float>(DEGREE(), float(i));

		// Do the reverse of making a difference table.
		for (int j = 0; j < DEGREE(); ++j)
		{
			controlPoints[j] = (float)coefficients[0];

			for (int i = 0; i < DEGREE(); ++i)
				coefficients[i] += coefficients[i + 1];
			printf("%0.2f, ", controlPoints[j]);
		}
		controlPoints[DEGREE()] = (float)coefficients[0];
		printf("%0.2f ]\n", controlPoints[DEGREE()]);
	}

	// Show the Bezier curve formula
	first = true;
	printf("\ny = f(t) = ");
	for (int degree = 0; degree <= DEGREE(); ++degree)
	{
		if (!first)
			printf(" + ");

		int bc = BinomialCoefficient<int>(DEGREE(), degree);
		if (bc != 1)
			printf("%i*", bc);
		printf("%0.2f", controlPoints[degree]);

		int degree1MinusT = DEGREE() - degree;
		int degreeT = degree;

		if (degree1MinusT > 1)
			printf("(1-t)^%i", degree1MinusT);
		else if (degree1MinusT == 1)
			printf("(1-t)");

		if (degreeT > 1)
			printf("t^%i", degreeT);
		else if (degreeT == 1)
			printf("t");

		first = false;
	}
	printf("\n");
}
