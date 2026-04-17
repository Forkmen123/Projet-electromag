# ---------------   AFFICHAGE potentiel --------------------------
x_slice = 1
V_2d  = potential[:, x_slice, :]   # shape (nz, ny)
Ey_2d = Ey[:, x_slice, :]
Ez_2d = Ez[:, x_slice, :]

y_axis = np.linspace(0, Ly,       ny)   # axe vertical   (court)
z_axis = np.linspace(0, L_finger, nz)   # axe horizontal (long)

# meshgrid : premier argument → colonnes (x du plot) = z
#            deuxième argument → lignes   (y du plot) = y
Z, Y = np.meshgrid(z_axis, y_axis)   # Z,Y shape (ny, nz)

# V doit aussi être (ny, nz) → transpose
V_plot  = V_2d.T    # (ny, nz)
Ey_plot = Ey_2d.T
Ez_plot = Ez_2d.T

plt.figure(figsize=(12, 4))
plt.imshow(
    V_plot,
    cmap="plasma",
    origin="lower",
    extent=[0, L_finger, 0, Ly],   # [z_min, z_max, y_min, y_max]
    aspect="equal",
)
plt.colorbar(label="Potentiel V")
plt.xlabel("z [µm]")
plt.ylabel("y [µm]")
plt.title(f"Potentiel — coupe x = {x_slice}")
plt.gca().set_aspect("equal", adjustable="box")

# ---------------   AFFICHAGE potentiel + gradient --------------------------
fig, ax = plt.subplots(figsize=(12, 4))

im = ax.contourf(Z, Y, V_plot, levels=60, cmap="plasma")
plt.colorbar(im, ax=ax, label="Potentiel V [V]")
ax.contour(Z, Y, V_plot, levels=15, colors="white", linewidths=0.5, alpha=0.6)

n_arrows = 30
step_z = max(1, nz // n_arrows)   # pas en colonnes (z)
step_y = max(1, ny // n_arrows)   # pas en lignes   (y)

ax.quiver(
    Z   [::step_y, ::step_z],     # x du plot = z
    Y   [::step_y, ::step_z],     # y du plot = y
    Ez_plot[::step_y, ::step_z],  # composante horizontale = Ez
    Ey_plot[::step_y, ::step_z],  # composante verticale   = Ey
    color="white", alpha=0.8,
    scale=None, width=0.003,
    headwidth=3, headlength=5,
)

ax.set_xlabel("z [µm]")
ax.set_ylabel("y [µm]")
ax.set_title(f"Potentiel et champ E — coupe x = {x_slice}")
ax.set_aspect("equal", adjustable="box")