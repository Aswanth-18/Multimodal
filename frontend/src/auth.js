import { initializeApp, getApps } from "firebase/app";
import {
  browserLocalPersistence,
  createUserWithEmailAndPassword,
  getAuth,
  GoogleAuthProvider,
  onAuthStateChanged,
  sendEmailVerification,
  sendPasswordResetEmail,
  setPersistence,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
  updateProfile,
} from "firebase/auth";

const config = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

export const firebaseConfigured = Object.values(config).every(Boolean);
let auth;
let pendingDoctorId = "";

export async function initializeFirebaseAuth() {
  if (!firebaseConfigured) {
    throw new Error("Firebase is not configured. Copy frontend/.env.example to frontend/.env.local and add your Firebase web-app values.");
  }
  const app = getApps().length ? getApps()[0] : initializeApp(config);
  auth = getAuth(app);
  await setPersistence(auth, browserLocalPersistence);
  return auth;
}

export function currentUser() {
  return auth?.currentUser ?? null;
}

export function watchAuth(callback) {
  return onAuthStateChanged(auth, callback);
}

export function rememberDoctorId(value) {
  pendingDoctorId = value.trim();
}

export function rememberedDoctorId() {
  return pendingDoctorId;
}

export async function registerWithEmail({ fullName, email, password }) {
  const credential = await createUserWithEmailAndPassword(auth, email, password);
  await updateProfile(credential.user, { displayName: fullName });
  await sendEmailVerification(credential.user);
  await signOut(auth);
}

export async function signInWithEmail(email, password) {
  return (await signInWithEmailAndPassword(auth, email, password)).user;
}

export async function signInWithGoogle() {
  const provider = new GoogleAuthProvider();
  provider.setCustomParameters({ prompt: "select_account" });
  return (await signInWithPopup(auth, provider)).user;
}

export function sendPasswordReset(email) {
  return sendPasswordResetEmail(auth, email);
}

export function logOut() {
  return signOut(auth);
}
