#version 430 core

in vec4 vcolor;
in vec3 vnormals;

out vec4 color;

void main()
{
    vec3 lightDirection = normalize(vec3(0.8, -0.5, 0.6));
    //normalColor = vec4(vnormals.x, vnormals.y, vnormals.z, 1.0);
    color = vec4(vcolor.rgb * max(0.0, dot(vnormals, -lightDirection)), vcolor.a);
}