////////////////////////////////////////////////////////////////////////
//
//   KAIST, Spring 2023
//   CS380: Introduction to Computer Graphics
//   Instructor: Minhyuk Sung (mhsung@kaist.ac.kr)
//   Last Update: Juil Koo (63days@kaist.ac.kr)
//
////////////////////////////////////////////////////////////////////////

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
// If your OS is LINUX, uncomment the line below.
//#include <tr1/memory>

#include <GL/glew.h>

#ifdef __MAC__

#   include <GLUT/glut.h>

#else
#   include <GL/glut.h>
#endif

#include "cvec.h"
#include "matrix4.h"
#include "rigtform.h"
#include "geometrymaker.h"
#include "ppm.h"
#include "glsupport.h"
#include "arcball.h"

using namespace std;      // for string, vector, iostream, and other standard C++ stuff
// If your OS is LINUX, uncomment the line below.
//using namespace tr1; // for shared_ptr

// G L O B A L S ///////////////////////////////////////////////////

// --------- IMPORTANT --------------------------------------------------------
// Before you start working on this assignment, set the following variable
// properly to indicate whether you want to use OpenGL 2.x with GLSL 1.0 or
// OpenGL 3.x+ with GLSL 1.3.
//
// Set g_Gl2Compatible = true to use GLSL 1.0 and g_Gl2Compatible = false to
// use GLSL 1.3. Make sure that your machine supports the version of GLSL you
// are using. In particular, on Mac OS X currently there is no way of using
// OpenGL 3.x with GLSL 1.3 when GLUT is used.
//
// If g_Gl2Compatible=true, shaders with -gl2 suffix will be loaded.
// If g_Gl2Compatible=false, shaders with -gl3 suffix will be loaded.
// To complete the assignment you only need to edit the shader files that get
// loaded
// ----------------------------------------------------------------------------
static const bool g_Gl2Compatible = true;


static const float g_frustMinFov = 60.0;  // A minimal of 60 degree field of view
static float g_frustFovY = g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)

static const float g_frustNear = -0.1;    // near plane
static const float g_frustFar = -50.0;    // far plane
static const float g_groundY = -2.0;      // y coordinate of the ground
static const float g_groundSize = 10.0;   // half the ground length

static int g_windowWidth = 512; // TODO: 512
static int g_windowHeight = 512; // TODO: 512
static bool g_mouseClickDown = false;    // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static int g_mouseClickX, g_mouseClickY; // coordinates for mouse click event
static int g_activeShader = 0;

struct ShaderState {
    GlProgram program;

    // Handles to uniform variables
    GLint h_uLight, h_uLight2;
    GLint h_uProjMatrix;
    GLint h_uModelViewMatrix;
    GLint h_uNormalMatrix;
    GLint h_uColor;

    // Handles to vertex attributes
    GLint h_aPosition;
    GLint h_aNormal;

    ShaderState(const char *vsfn, const char *fsfn) {
      readAndCompileShader(program, vsfn, fsfn);

      const GLuint h = program; // short hand

      // Retrieve handles to uniform variables
      h_uLight = safe_glGetUniformLocation(h, "uLight");
      h_uLight2 = safe_glGetUniformLocation(h, "uLight2");
      h_uProjMatrix = safe_glGetUniformLocation(h, "uProjMatrix");
      h_uModelViewMatrix = safe_glGetUniformLocation(h, "uModelViewMatrix");
      h_uNormalMatrix = safe_glGetUniformLocation(h, "uNormalMatrix");
      h_uColor = safe_glGetUniformLocation(h, "uColor");

      // Retrieve handles to vertex attributes
      h_aPosition = safe_glGetAttribLocation(h, "aPosition");
      h_aNormal = safe_glGetAttribLocation(h, "aNormal");

      if (!g_Gl2Compatible)
        glBindFragDataLocation(h, 0, "fragColor");
      checkGlErrors();
    }

};

static const int g_numShaders = 2;
static const char *const g_shaderFiles[g_numShaders][2] = {
  {"./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader"},
  {"./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader"}
};
static const char *const g_shaderFilesGl2[g_numShaders][2] = {
  {"./shaders/basic-gl2.vshader", "./shaders/diffuse-gl2.fshader"},
  {"./shaders/basic-gl2.vshader", "./shaders/solid-gl2.fshader"}
};
static vector<shared_ptr<ShaderState> > g_shaderStates; // our global shader states

// --------- Geometry

// Macro used to obtain relative offset of a field within a struct
#define FIELD_OFFSET(StructType, field) &(((StructType *)0)->field)

// A vertex with floating point position and normal
struct VertexPN {
    Cvec3f p, n;

    VertexPN() {}

    VertexPN(float x, float y, float z,
             float nx, float ny, float nz)
      : p(x, y, z), n(nx, ny, nz) {}

    // Define copy constructor and assignment operator from GenericVertex so we can
    // use make* functions from geometrymaker.h
    VertexPN(const GenericVertex &v) {
      *this = v;
    }

    VertexPN &operator=(const GenericVertex &v) {
      p = v.pos;
      n = v.normal;
      return *this;
    }
};

struct Geometry {
    GlBufferObject vbo, ibo;
    int vboLen, iboLen;

    Geometry(VertexPN *vtx, unsigned short *idx, int vboLen, int iboLen) {
      this->vboLen = vboLen;
      this->iboLen = iboLen;

      // Now create the VBO and IBO
      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glBufferData(GL_ARRAY_BUFFER, sizeof(VertexPN) * vboLen, vtx, GL_STATIC_DRAW);

      // index buffer object
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * iboLen, idx, GL_STATIC_DRAW);
    }

    void draw(const ShaderState &curSS) {
      // Enable the attributes used by our shader
      safe_glEnableVertexAttribArray(curSS.h_aPosition);
      safe_glEnableVertexAttribArray(curSS.h_aNormal);

      // bind vbo
      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      safe_glVertexAttribPointer(curSS.h_aPosition, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, p));
      safe_glVertexAttribPointer(curSS.h_aNormal, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, n));

      // bind ibo
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

      // draw!
      glDrawElements(GL_TRIANGLES, iboLen, GL_UNSIGNED_SHORT, 0);

      // Disable the attributes used by our shader
      safe_glDisableVertexAttribArray(curSS.h_aPosition);
      safe_glDisableVertexAttribArray(curSS.h_aNormal);
    }
};


// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube1, g_cube2, g_arcball;

// --------- Scene

static const Cvec3 g_light1(2.0, 3.0, 14.0), g_light2(-2, -3.0, -5.0);  // define two lights positions in world space
static RigTForm g_skyRbt = RigTForm(Cvec3(0.0, 0.25, 4.0));
static RigTForm g_objectRbt[2] = {
  RigTForm(Cvec3(0.75, 0, 0)),
  RigTForm(Cvec3(-0.75, 0, 0)),
};
static Cvec3f g_objectColors[2] = {
  Cvec3f(1, 0, 0),
  Cvec3f(0, 0, 1),
};
static Cvec3f g_arcballColor = Cvec3f(0, 1, 0);

static RigTForm g_currentObjectRbt = g_objectRbt[0];

static RigTForm g_currentEyeRbt = RigTForm(Cvec3(0.0, 0.25, 4.0));
static int g_currentEyeIdx = 0; // 0: sky, 1: cube1, 2: cube2
static string g_eyeNames[3] = {"sky", "cube1", "cube2"};

static int g_manipulatedObject[3] = {
  -1,
  0,
  1,
};
static int g_currentManipulatedObjectIdx = 1; // 0: sky, 1: cube1, 2: cube2
static string g_manipulatedObjectNames[3] = {"sky", "cube1", "cube2"};

static int g_currentSkyFrame = 0; // 0: world-sky, 1: sky-sky
static string g_skyFrameNames[2] = {"world-sky", "sky-sky"};

static double g_arcballScreenRadius = 0.25 * min(g_windowWidth, g_windowHeight);
static double g_arcballScale = 0.01;

///////////////// END OF G L O B A L S //////////////////////////////////////////////////


static void initGround() {
  // A x-z plane at y = g_groundY of dimension [-g_groundSize, g_groundSize]^2
  VertexPN vtx[4] = {
    VertexPN(-g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
    VertexPN(-g_groundSize, g_groundY, g_groundSize, 0, 1, 0),
    VertexPN(g_groundSize, g_groundY, g_groundSize, 0, 1, 0),
    VertexPN(g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
  };
  unsigned short idx[] = {0, 1, 2, 0, 2, 3};
  g_ground.reset(new Geometry(&vtx[0], &idx[0], 4, 6));
}

static void initCubes() {
  int ibLen, vbLen;
  getCubeVbIbLen(vbLen, ibLen);

  // Temporary storage for cube geometry
  vector<VertexPN> vtx(vbLen);
  vector<unsigned short> idx(ibLen);

  makeCube(1, vtx.begin(), idx.begin());
  g_cube1.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  // Temporary storage for cube geometry
  vector<VertexPN> vtx2(vbLen);
  vector<unsigned short> idx2(ibLen);

  makeCube(1, vtx2.begin(), idx2.begin());
  g_cube2.reset(new Geometry(&vtx2[0], &idx2[0], vbLen, ibLen));
}

static void initArcball() {
  int ibLen, vbLen;
  getSphereVbIbLen(10, 10, vbLen, ibLen);

  vector<VertexPN> vtx(vbLen);
  vector<unsigned short> idx(ibLen);

  makeSphere(1, 10, 10, vtx.begin(), idx.begin());
  g_arcball.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));
}

// -- 이건 다음주나 배움
// takes a projection matrix and send to the the shaders
static void sendProjectionMatrix(const ShaderState &curSS, const Matrix4 &projMatrix) {
  GLfloat glmatrix[16];
  projMatrix.writeToColumnMajorMatrix(glmatrix); // send projection matrix
  safe_glUniformMatrix4fv(curSS.h_uProjMatrix, glmatrix);
}

// -- 이건 써먹어야함
// takes MVM and its normal matrix to the shaders
static void sendModelViewNormalMatrix(const ShaderState &curSS, const Matrix4 &MVM, const Matrix4 &NMVM) {
  GLfloat glmatrix[16];
  MVM.writeToColumnMajorMatrix(glmatrix); // send MVM
  safe_glUniformMatrix4fv(curSS.h_uModelViewMatrix, glmatrix);

  NMVM.writeToColumnMajorMatrix(glmatrix); // send NMVM
  safe_glUniformMatrix4fv(curSS.h_uNormalMatrix, glmatrix);
}

// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
  if (g_windowWidth >= g_windowHeight)
    g_frustFovY = g_frustMinFov;
  else {
    const double RAD_PER_DEG = 0.5 * CS380_PI / 180;
    g_frustFovY =
      atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight / g_windowWidth, cos(g_frustMinFov * RAD_PER_DEG)) /
      RAD_PER_DEG;
  }
}

static Matrix4 makeProjectionMatrix() {
  return Matrix4::makeProjection(
    g_frustFovY, g_windowWidth / static_cast <double> (g_windowHeight),
    g_frustNear, g_frustFar);
}

static void drawStuff() {
  // short hand for current shader state
  const ShaderState &curSS = *g_shaderStates[g_activeShader];

  // build & send proj. matrix to vshader
  const Matrix4 projmat = makeProjectionMatrix();
  sendProjectionMatrix(curSS, projmat);

  // use the skyRbt as the eyeRbt (스카이캠) Rbt body transformation
  // -- 과제하기 위해 eye frame을 바꿔야함
  if (g_currentEyeIdx == 0) {
    g_currentEyeRbt = g_skyRbt;
  } else if (g_currentEyeIdx == 1) {
    g_currentEyeRbt = g_objectRbt[0];
  } else {
    g_currentEyeRbt = g_objectRbt[1];
  }
  const RigTForm eyeRbt = g_currentEyeRbt;
  cout << "eyeRbt x: " << eyeRbt.getTranslation()[0] << ", y: " << eyeRbt.getTranslation()[1] << ", z: " << eyeRbt.getTranslation()[2] << endl;
//  cout << "debug12" << endl;
  const RigTForm invEyeRbt = inv(eyeRbt);
//  cout << "debug13" << endl;

  const Cvec3 eyeLight1 = Cvec3(invEyeRbt * Cvec4(g_light1, 1)); // g_light1 position in eye coordinates
  const Cvec3 eyeLight2 = Cvec3(invEyeRbt * Cvec4(g_light2, 1)); // g_light2 position in eye coordinates
  safe_glUniform3f(curSS.h_uLight, eyeLight1[0], eyeLight1[1], eyeLight1[2]);
  safe_glUniform3f(curSS.h_uLight2, eyeLight2[0], eyeLight2[1], eyeLight2[2]);
//  cout << "debug14" << endl;

  // draw ground
  // ===========
  const RigTForm groundRbt = RigTForm();  // identity
  Matrix4 MVM = rigTFormToMatrix(invEyeRbt * groundRbt); // groundRbt ground frame, MVM (model view matrix) E^-1O
  Matrix4 NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  safe_glUniform3f(curSS.h_uColor, 0.1, 0.95, 0.1); // set color
  g_ground->draw(curSS);
//  cout << "debug15" << endl;

  // draw cubes
  // ==========
  MVM = rigTFormToMatrix(invEyeRbt * g_objectRbt[0]);
//  cout << "debug16A" << endl;
  NMVM = normalMatrix(MVM);
//  cout << "debug16B" << endl;
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
//  cout << "debug16C" << endl;
  safe_glUniform3f(curSS.h_uColor, g_objectColors[0][0], g_objectColors[0][1], g_objectColors[0][2]);
//  cout << "debug16D" << endl;
  g_cube1->draw(curSS);
//  cout << "debug16E" << endl;

  MVM = rigTFormToMatrix(invEyeRbt * g_objectRbt[1]);
//  cout << "debug17A" << endl;
  NMVM = normalMatrix(MVM);
//  cout << "debug17B" << endl;
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
//  cout << "debug17C" << endl;
  safe_glUniform3f(curSS.h_uColor, g_objectColors[1][0], g_objectColors[1][1], g_objectColors[1][2]);
//  cout << "debug17D" << endl;
  g_cube2->draw(curSS);
//  cout << "debug17E" << endl;

  // draw arcball
  // ============
  string manipulatedObjectName = g_manipulatedObjectNames[g_currentManipulatedObjectIdx];
  string currentEyeName = g_eyeNames[g_currentEyeIdx];
  string currentSkyFrame = g_skyFrameNames[g_currentSkyFrame];
  if (
    (manipulatedObjectName == "cube1" && currentEyeName != "cube1")
    || (manipulatedObjectName == "cube2" && currentEyeName != "cube2")
    || (manipulatedObjectName == "sky" && currentEyeName == "sky" && currentSkyFrame == "world-sky")
  ) {
    double z;
    if (manipulatedObjectName == "sky") {
      // world-sky 프레임이라고 가정할 수 있음
      z = (invEyeRbt * RigTForm(Cvec3(0, 0, 0))).getTranslation()[2];
    } else {
      // cube
      z = (invEyeRbt * g_objectRbt[g_manipulatedObject[g_currentManipulatedObjectIdx]]).getTranslation()[2];
    }

    // 눌려있는 동안 업데이트 X
    if (!(g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton))) {
      g_arcballScale = getScreenToEyeScale(z, g_frustFovY, g_windowHeight);
    }

    double radius = g_arcballScale * g_arcballScreenRadius;
    Matrix4 scale = Matrix4::makeScale(Cvec3(1, 1, 1)  * radius);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    if (manipulatedObjectName == "sky") {
      // world-sky 프레임이라고 가정할 수 있음
      MVM = rigTFormToMatrix(invEyeRbt * RigTForm(Cvec3(0, 0, 0))) * scale;
    } else {
      // cube
      MVM = rigTFormToMatrix(invEyeRbt * g_objectRbt[g_manipulatedObject[g_currentManipulatedObjectIdx]]) * scale;
    }
    NMVM = normalMatrix(MVM);
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    safe_glUniform3f(curSS.h_uColor, g_arcballColor[0], g_arcballColor[1], g_arcballColor[2]);
    g_arcball->draw(curSS);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//    cout << "debug18" << endl;
  }
}

static void display() {
  glUseProgram(g_shaderStates[g_activeShader]->program);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);                   // clear framebuffer color&depth

  drawStuff();

  glutSwapBuffers();                                    // show the back buffer (where we rendered stuff)

  checkGlErrors();
}

static void reshape(const int w, const int h) {
  g_windowWidth = w;
  g_windowHeight = h;
  g_arcballScreenRadius = 0.25 * min(g_windowWidth, g_windowHeight);
  glViewport(0, 0, w, h);
  cerr << "Size of window is now " << w << "x" << h << endl;
  cerr << "Arcball Screen Radius is now " << g_arcballScreenRadius <<  endl;
  updateFrustFovY();
  glutPostRedisplay();
}

Cvec3 getArcballPoint(Cvec2 screenPoint, Cvec3 center, double radius, Cvec3 eye) {
  cout << "getArcballPoint!!" << endl;

  Cvec3 ray = getModelViewRay(screenPoint, g_frustFovY, g_windowWidth, g_windowHeight);
  cout << "ray vector before x: " << ray[0] << ", y: " << ray[1] << ", z: " << ray[2] << endl;
  ray = Cvec3(-ray[0], -ray[1], -ray[2]);
  cout << "ray vector after x: " << ray[0] << ", y: " << ray[1] << ", z: " << ray[2] << endl;

  cout << "screenPoint x: " << screenPoint[0] << ", y: " << screenPoint[1] << endl;
  Cvec3 centerRay = center - eye;
  cout << "radius: " << radius << endl;

  cout << "eye point x: " << eye[0] << ", y: " << eye[1] << ", z: " << eye[2] << endl;
  cout << "center point x: " << center[0] << ", y: " << center[1] << ", z: " << center[2] << endl;
  cout << "eye->center vector x: " << centerRay[0] << ", y: " << centerRay[1] << ", z: " << centerRay[2] << endl;

  double a = norm2(ray);
  cout << "a: " << a << endl;
  double b = -2 * dot(ray, centerRay);
  cout << "b: " << b << endl;
  double c = norm2(centerRay) - radius * radius;
  cout << "c: " << c << endl;
  double d = b * b - 4 * a * c;
  cout << "b^2 - 4ac: " << d << endl;
  bool intersect = b * b - 4 * a * c >= 0;
  cout << "intersect: " << intersect << endl;

  double root;
  if (intersect) {
    root = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
    cout << "root1: " << root << endl;
  } else {
    root = -b / (2 * a);
    cout << "not intersect! root2" << endl;
  }
  Cvec3 point = eye + ray * root;
  double distanceOfCenter = sqrt((point[0] - center[0]) * (point[0] - center[0]) + (point[1] - center[1]) * (point[1] - center[1]) + (point[2] - center[2]) * (point[2] - center[2]));
  cout << "distanceOfCenter: " << distanceOfCenter << endl;
  if (abs(distanceOfCenter - radius) > CS380_EPS) {
    point = center + (point - center) * radius / distanceOfCenter;
    distanceOfCenter = sqrt((point[0] - center[0]) * (point[0] - center[0]) + (point[1] - center[1]) * (point[1] - center[1]) + (point[2] - center[2]) * (point[2] - center[2]));
    cout << "new distanceOfCenter: " << distanceOfCenter << endl;
  }
  cout << "point x: " << point[0] << ", y: " << point[1] << ", z: " << point[2] << endl;
  return point;
}

static void motion(const int x, const int y) {
  cout << "motion! x:" << x << " y:" << y << endl;

  const double dx = x - g_mouseClickX;
  const double dy = g_windowHeight - y - 1 - g_mouseClickY;
  cout << "dx: " << dx << ", dy: " << dy << endl;

  RigTForm auxFrame;
  if (g_currentEyeIdx == 0) {
    g_currentEyeRbt = g_skyRbt;
  } else if (g_currentEyeIdx == 1) {
    g_currentEyeRbt = g_objectRbt[0];
  } else {
    g_currentEyeRbt = g_objectRbt[1];
  }

  bool arcballRotation = false;
  RigTForm arcballRotationRbt = RigTForm();
  // arcball 회전
  string manipulatedObjectName = g_manipulatedObjectNames[g_currentManipulatedObjectIdx];
  string currentEyeName = g_eyeNames[g_currentEyeIdx];
  string currentSkyFrame = g_skyFrameNames[g_currentSkyFrame];
  if (
    (manipulatedObjectName == "cube1" && currentEyeName != "cube1")
    || (manipulatedObjectName == "cube2" && currentEyeName != "cube2")
    || (manipulatedObjectName == "sky" && currentEyeName == "sky" && currentSkyFrame == "world-sky")
  ) {
    Cvec3 centerOfArcball;
    if (manipulatedObjectName == "sky") {
      // world-sky 프레임이라고 가정할 수 있음
      centerOfArcball = Cvec3(0, 0, 0);
    } else {
      // cube
      centerOfArcball = g_objectRbt[g_manipulatedObject[g_currentManipulatedObjectIdx]].getTranslation();
    }

    double radius = g_arcballScale * g_arcballScreenRadius;
    cout << "radius: " << radius << endl;
    cout << "scale: " << g_arcballScale << endl;
    cout << "screenRadius: " << g_arcballScreenRadius << endl;
    Cvec3 eye = g_currentEyeRbt.getTranslation();
    Cvec3 arcballPoint0 = getArcballPoint(Cvec2(g_mouseClickX, g_mouseClickY), centerOfArcball, radius, eye);
    cout << "arcballPoint0 x: " << arcballPoint0[0] << ", y: " << arcballPoint0[1] << ", z: " << arcballPoint0[2] << endl;
    double distanceOfCenter0 = sqrt((arcballPoint0[0] - centerOfArcball[0]) * (arcballPoint0[0] - centerOfArcball[0]) + (arcballPoint0[1] - centerOfArcball[1]) * (arcballPoint0[1] - centerOfArcball[1]) + (arcballPoint0[2] - centerOfArcball[2]) * (arcballPoint0[2] - centerOfArcball[2]));
    cout << "distanceOfCenter0: " << distanceOfCenter0 << endl;

    Cvec3 arcballPoint1 = getArcballPoint(Cvec2(g_mouseClickX + dx, g_mouseClickY + dy), centerOfArcball, radius, eye);;
    cout << "arcballPoint1 x: " << arcballPoint1[0] << ", y: " << arcballPoint1[1] << ", z: " << arcballPoint1[2] << endl;

    double distanceOfCenter1 = sqrt((arcballPoint1[0] - centerOfArcball[0]) * (arcballPoint1[0] - centerOfArcball[0]) + (arcballPoint1[1] - centerOfArcball[1]) * (arcballPoint1[1] - centerOfArcball[1]) + (arcballPoint1[2] - centerOfArcball[2]) * (arcballPoint1[2] - centerOfArcball[2]));
    cout << "distanceOfCenter1: " << distanceOfCenter1 << endl;

      Cvec3 arcballVector0 = normalize(arcballPoint0 - centerOfArcball);
      Cvec3 arcballVector1 = normalize(arcballPoint1 - centerOfArcball);
      Quat quatVector0 = Quat(0, arcballVector0);
  //    cout << "quatVector0 i: " << arcballVector0[1] << ", j: " << arcballVector0[2] << ", k: " << arcballVector0[3] << ", w: " << arcballVector0[0] << endl;
      Quat quatVector1 = Quat(0, arcballVector1);
  //    cout << "quatVector1 i: " << arcballVector1[1] << ", j: " << arcballVector1[2] << ", k: " << arcballVector1[3] << ", w: " << arcballVector1[0] << endl;

  //    cout << "arcballRotation: true" << endl;
      arcballRotation = true;
      if (manipulatedObjectName == "sky") {
        // world-sky 프레임이라고 가정할 수 있음
        arcballRotationRbt.setRotation(quatVector1 * (quatVector0 * -1));
        arcballRotationRbt = inv(arcballRotationRbt);
      } else {
        // cube
        arcballRotationRbt.setRotation(quatVector1 * (quatVector0 * -1));
      }
  }

  if (g_manipulatedObjectNames[g_currentManipulatedObjectIdx] == "sky") {
    // object가 sky
    cout << "object is sky!!" << endl;
    if (g_eyeNames[g_currentEyeIdx] != "sky") {
      // eye frame이 cube인 경우
      return;
    } else if (g_skyFrameNames[g_currentSkyFrame] == "world-sky") {
      // world-sky
      cout << "eye is world-sky!!" << endl;
      auxFrame = linFact(g_skyRbt);
    } else {
      // sky-sky
      cout << "eys is sky-sky!!" << endl;
      auxFrame = transFact(g_skyRbt) * linFact(g_skyRbt);
    }
  } else {
    // object가 cube
    cout << "object is cube!!" << endl;
    auxFrame = transFact(g_objectRbt[g_manipulatedObject[g_currentManipulatedObjectIdx]]) * linFact(g_currentEyeRbt);
  }

  RigTForm m;
  if (g_mouseLClickButton && !g_mouseRClickButton) { // left button down?
    if (arcballRotation) {
      m = arcballRotationRbt;
    } else {
      if (g_manipulatedObjectNames[g_currentManipulatedObjectIdx] != "sky"
        && g_manipulatedObjectNames[g_currentManipulatedObjectIdx] != g_eyeNames[g_currentEyeIdx]
      ) {
        m = RigTForm(Quat::makeXRotation(-dy) * Quat::makeYRotation(dx));
      } else {
        m = RigTForm(Quat::makeXRotation(dy) * Quat::makeYRotation(-dx));
      }
    }
  } else if (g_mouseRClickButton && !g_mouseLClickButton) { // right button down?
//    cout << "g_arcballScale1: " << g_arcballScale << endl;
    // world-sky이면서, object, eye 둘다 sky
    if (g_eyeNames[g_currentEyeIdx] == "sky"
      && g_manipulatedObjectNames[g_currentManipulatedObjectIdx] == "sky"
      && g_skyFrameNames[g_currentSkyFrame] == "world-sky"
    ) {
      m = RigTForm(Cvec3(-dx, -dy, 0) * g_arcballScale);
    } else {
      m = RigTForm(Cvec3(dx, dy, 0) * g_arcballScale);
    }
  } else if (g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton)) {  // middle or (left and right) button down?
//    cout << "g_arcballScale2: " << g_arcballScale << endl;
//    cout << "debugxxxxxx" << endl;
    m = RigTForm(Cvec3(0, 0, -dy) * g_arcballScale);
//    g_arcballScreenRadius = (g_arcballScale / 0.01);
  }

  // AMA^-1
//  cout << "debug10" << endl;
  RigTForm ama = auxFrame * m * inv(auxFrame);
//  cout << "debug11" << endl;

  if (g_mouseClickDown) {
    if (g_manipulatedObjectNames[g_currentManipulatedObjectIdx] == "sky") {
      cout << "hello" << endl;
      g_skyRbt = ama * g_skyRbt;
    } else {
      g_objectRbt[g_manipulatedObject[g_currentManipulatedObjectIdx]] = ama * g_objectRbt[g_manipulatedObject[g_currentManipulatedObjectIdx]];
    }
    glutPostRedisplay(); // we always redraw if we changed the scene
  }

  g_mouseClickX = x;
  g_mouseClickY = g_windowHeight - y - 1;
}


static void mouse(const int button, const int state, const int x, const int y) {
  cout << "mouse! button:" << button << " state:" << state << " x:" << x << " y:" << y << endl;
  g_mouseClickX = x;
  g_mouseClickY =
    g_windowHeight - y - 1;  // conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system

  g_mouseLClickButton |= (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN);
  g_mouseRClickButton |= (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN);
  g_mouseMClickButton |= (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN);

  g_mouseLClickButton &= !(button == GLUT_LEFT_BUTTON && state == GLUT_UP);
  g_mouseRClickButton &= !(button == GLUT_RIGHT_BUTTON && state == GLUT_UP);
  g_mouseMClickButton &= !(button == GLUT_MIDDLE_BUTTON && state == GLUT_UP);

  g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;

  // 마우스 떼졌을때 redraw
  glutPostRedisplay();
}


static void keyboard(const unsigned char key, const int x, const int y) {
  switch (key) {
    case 27:
      exit(0);                                  // ESC
    case 'h':
      cout << " ============== H E L P ==============\n\n"
           << "h\t\thelp menu\n"
           << "s\t\tsave screenshot\n"
           << "f\t\tToggle flat shading on/off.\n"
           << "o\t\tCycle object to edit\n"
           << "v\t\tCycle view\n"
           << "drag left mouse to rotate\n" << endl;
      break;
    case 's':
      glFlush();
      writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
      break;
    case 'f':
      g_activeShader ^= 1;
      break;
    case 'v':
      g_currentEyeIdx = (g_currentEyeIdx + 1) % 3;
      cout << "현재 eye: " << g_eyeNames[g_currentEyeIdx] << endl;
      break;
    case 'o':
      g_currentManipulatedObjectIdx = (g_currentManipulatedObjectIdx + 1) % 3;
      cout << "현재 object: " << g_manipulatedObjectNames[g_currentManipulatedObjectIdx] << endl;
      break;
    case 'm':
      if (g_manipulatedObjectNames[g_currentManipulatedObjectIdx] == "sky"
        && g_eyeNames[g_currentEyeIdx] == "sky"
      ) {
        g_currentSkyFrame = (g_currentSkyFrame + 1) % 2;
        cout << "현재 sky frame: " << g_skyFrameNames[g_currentSkyFrame] << endl;
      } else {
        cout << "m 옵션은 eye랑 object가 둘다 sky일 때만 가능합니다." << endl;
      }
  }
  glutPostRedisplay();
}

static void initGlutState(int argc, char *argv[]) {
  glutInit(&argc, argv);                                  // initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);  //  RGBA pixel channels and double buffering
  glutInitWindowSize(g_windowWidth, g_windowHeight);      // create a window
  glutCreateWindow("CS380: Assignment 2");                       // title the window

  glutDisplayFunc(display);                               // display rendering callback
  glutReshapeFunc(reshape);                               // window reshape callback
  glutMotionFunc(motion);                                 // mouse movement callback
  glutMouseFunc(mouse);                                   // mouse click callback
  glutKeyboardFunc(keyboard);
}

static void initGLState() {
  glClearColor(128. / 255., 200. / 255., 255. / 255., 0.);
  glClearDepth(0.);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_GREATER);
  glReadBuffer(GL_BACK);
  if (!g_Gl2Compatible)
    glEnable(GL_FRAMEBUFFER_SRGB);
}

static void initShaders() {
  g_shaderStates.resize(g_numShaders);
  for (int i = 0; i < g_numShaders; ++i) {
    if (g_Gl2Compatible)
      g_shaderStates[i].reset(new ShaderState(g_shaderFilesGl2[i][0], g_shaderFilesGl2[i][1]));
    else
      g_shaderStates[i].reset(new ShaderState(g_shaderFiles[i][0], g_shaderFiles[i][1]));
  }
}

static void initGeometry() {
  initGround();
  initCubes();
  initArcball();
}

int main(int argc, char *argv[]) {
  try {
    initGlutState(argc, argv);

    glewInit(); // load the OpenGL extensions

    cout << (g_Gl2Compatible ? "Will use OpenGL 2.x / GLSL 1.0" : "Will use OpenGL 3.x / GLSL 1.3") << endl;
    if ((!g_Gl2Compatible) && !GLEW_VERSION_3_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.3");
    else if (g_Gl2Compatible && !GLEW_VERSION_2_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.0");

    initGLState();
    initShaders();
    initGeometry();

    glutMainLoop();
    return 0;
  }
  catch (const runtime_error &e) {
    cout << "Exception caught: " << e.what() << endl;
    return -1;
  }
}
