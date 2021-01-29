settings.outformat = "pdf";
settings.prc = false;
settings.render = 48;
import three;

unitsize(1cm);

draw(O -- 4X, arrow=Arrow3(), L=Label("$x$", position=EndPoint, align=W));
draw(O -- 4Y, arrow=Arrow3(), L=Label("$y$", position=EndPoint, align=E));
draw(O -- 4Z, arrow=Arrow3(), L=Label("$z$", position=EndPoint, align=N));

triple camera = (0, 2, 3);
real canvas_distance = 3.0;
triple cube_center = (10, 0, 0);
triple view_direction = cube_center - camera;
view_direction = unit(view_direction);
triple canvas_center = camera + canvas_distance * view_direction;

triple horizon = (0, -1, 0);
triple proj = view_direction * dot(view_direction, horizon) / dot(view_direction, view_direction);
horizon = horizon - proj;
triple dir_x = horizon / length(horizon);
triple dir_y = cross(dir_x, view_direction);
dir_y = unit(dir_y);

pair angles_h = (radians(-29), radians(29));
pair angles_v = (radians(-21), radians(21));

int img_width = 10;
int img_height = 7;

triple left_arm = tan(angles_h.x) * canvas_distance * dir_x;
triple right_arm = tan(angles_h.y) * canvas_distance * dir_x;
triple top_arm = tan(angles_v.y) * canvas_distance * dir_y;
triple bottom_arm = tan(angles_v.x) * canvas_distance * dir_y;

triple canvas_lb_corner = canvas_center + left_arm + bottom_arm;
triple canvas_dir_1 = 2 * right_arm;
triple canvas_dir_2 = 2 * top_arm;
path3 canvas = plane(O=canvas_lb_corner, canvas_dir_1, canvas_dir_2);

triple baby_step_x = (right_arm - left_arm) / img_width;
triple baby_step_y = (bottom_arm - top_arm) / img_height;
triple canvas_lu_corner = canvas_center + left_arm + top_arm;
for (int i = 1; i < img_height; ++i)
{
    triple start = canvas_lu_corner + i * baby_step_y;
    triple end = start + (right_arm - left_arm);
    draw(start -- end, blue);
}

for (int j = 1; j < img_width; ++j)
{
    triple start = canvas_lu_corner + j * baby_step_x;
    triple end = start + (bottom_arm - top_arm);
    draw(start -- end, blue);
}

real edge_length = 3.2;
triple cube_corner_0 = cube_center - 0.5 * edge_length * (X+Y+Z);
triple cube_corner_1 = cube_center + 0.5 * edge_length * (X+Y+Z);

path3[] cube = box(cube_corner_0, cube_corner_1);

triple example_point = cube_center - 0.5 * edge_length * X + 0.34 * Z + 0.40 * Y;
dot(example_point);
draw(camera -- example_point, arrow=Arrow3());

dot(camera);
// dot(cube_center);
// dot(canvas_center);
draw(cube, blue);
draw(canvas, blue);