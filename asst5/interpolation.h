#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include "rigtform.h"

namespace Interpolation {

	// Linear interpolation of two coordinate vectors
	inline Cvec3 lerp(const Cvec3& c0, const Cvec3& c1, const double& alpha) {
    double resultX = (1 - alpha) * c0[0] + alpha * c1[0];
    double resultY = (1 - alpha) * c0[1] + alpha * c1[1];
    double resultZ = (1 - alpha) * c0[2] + alpha * c1[2];
    return Cvec3(resultX, resultY, resultZ);
	}

	// Spherical interpolation of two quaternions
	inline Quat slerp(const Quat& q0, const Quat& q1, const double& alpha) {
//    Quat base = q1 * inv(q0);
//    // conditionally negate
//    if (base[0] < 0) {
//      base = base * -1;
//    }

    // calculate quatAngle
    double cosValue = dot(q0,q1);
    double sinValue = norm(cross(Cvec3(q0[1], q0[2], q0[3]), Cvec3(q1[1], q1[2], q1[3])));

    double quatAngle = atan2(sinValue, cosValue);
    return q0.operator*(sin((1 - alpha) * quatAngle) / sin(quatAngle)) + q1.operator*(sin(alpha *quatAngle) / sin(quatAngle));
	}

	// Linear interpolation of two RigTForms
	// Note: you should use the lerp and slerp functions you implemented above
	inline RigTForm Linear(const RigTForm& rbt0, const RigTForm& rbt1, const double& alpha) {
    Cvec3 t = lerp(rbt0.getTranslation(), rbt1.getTranslation(), alpha);
    Quat r = slerp(rbt0.getRotation(), rbt1.getRotation(), alpha);
		return RigTForm(t, r);	// Replace this value with your own code
	}

}
#endif